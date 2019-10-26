/*
 * ElastoPlasticityTest.cpp
 *
 *  Created on: Sep 30, 2019
 */

#include "Teuchos_UnitTestHarness.hpp"

#include "PlatoTestHelpers.hpp"

#include <memory>
#include <ostream>

#include <Omega_h_mesh.hpp>
#include <Omega_h_assoc.hpp>

#include "plato/Simp.hpp"
#include "plato/Simplex.hpp"
#include "plato/Kinetics.hpp"
#include "plato/BodyLoads.hpp"
#include "plato/ParseTools.hpp"
#include "plato/NaturalBCs.hpp"
#include "plato/ScalarGrad.hpp"
#include "plato/Projection.hpp"
#include "plato/WorksetBase.hpp"
#include "plato/EssentialBCs.hpp"
#include "plato/ProjectToNode.hpp"
#include "plato/FluxDivergence.hpp"
#include "plato/SimplexFadTypes.hpp"
#include "plato/StressDivergence.hpp"
#include "plato/SimplexPlasticity.hpp"
#include "plato/VectorFunctionVMS.hpp"
#include "plato/PlatoStaticsTypes.hpp"
#include "plato/ScalarFunctionBase.hpp"
#include "plato/PressureDivergence.hpp"
#include "plato/StabilizedMechanics.hpp"
#include "plato/PlatoAbstractProblem.hpp"
#include "plato/Plato_TopOptFunctors.hpp"
#include "plato/InterpolateFromNodal.hpp"
#include "plato/LinearElasticMaterial.hpp"
#include "plato/ScalarFunctionIncBase.hpp"
#include "plato/J2PlasticityUtilities.hpp"
#include "plato/LocalVectorFunctionInc.hpp"
#include "plato/ThermoPlasticityUtilities.hpp"
#include "plato/LinearTetCubRuleDegreeOne.hpp"

#include "plato/ImplicitFunctors.hpp"
#include "plato/Plato_Solve.hpp"
#include "plato/ApplyConstraints.hpp"
#include "plato/ScalarFunctionBaseFactory.hpp"
#include "plato/ScalarFunctionIncBaseFactory.hpp"

#include "KokkosBatched_LU_Decl.hpp"
#include "KokkosBatched_LU_Serial_Impl.hpp"
#include "KokkosBatched_Trsm_Decl.hpp"
#include "KokkosBatched_Trsm_Serial_Impl.hpp"

#include <Kokkos_Concepts.hpp>
#include "KokkosKernels_SparseUtils.hpp"
#include "KokkosSparse_spgemm.hpp"
#include "KokkosSparse_spadd.hpp"
#include "KokkosSparse_CrsMatrix.hpp"
#include <KokkosKernels_IOUtils.hpp>

#include <Teuchos_XMLParameterListCoreHelpers.hpp>

namespace Plato
{

/******************************************************************************//**
 * \brief Flatten vector workset.  Takes 2D view and converts it into a 1D view.
 *
 * \tparam NumLocalDofsPerCell number of local degrees of freedom per cell
 * \tparam AViewType Input workset, as a 2-D Kokkos::View
 * \tparam BViewType Output workset, as a 1-D Kokkos::View
 *
 * \param [in] aNumCells number of cells, i.e. elements
 * \param [in] aInput input workset (NumCells, LocalNumCellDofs)
 * \param [in/out] aOutput output vector (NumCells * LocalNumCellDofs)
**********************************************************************************/
template<Plato::OrdinalType NumLocalDofsPerCell, class AViewType, class BViewType>
inline void flatten_vector_workset(const Plato::OrdinalType& aNumCells,
                                   AViewType& aInput,
                                   BViewType& aOutput)
{
    if(aInput.size() <= static_cast<Plato::OrdinalType>(0))
    {
        THROWERR("\nInput Kokkos::View is empty, i.e. size <= 0.\n")
    }
    if(aOutput.size() <= static_cast<Plato::OrdinalType>(0))
    {
        THROWERR("\nOutput Kokkos::View is empty, i.e. size <= 0.\n")
    }
    if(aNumCells <= static_cast<Plato::OrdinalType>(0))
    {
        THROWERR("\nNumber of cells, i.e. elements, argument is <= zero.\n");
    }

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, aNumCells),LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        const auto tDofOffset = aCellOrdinal * NumLocalDofsPerCell;
        for (Plato::OrdinalType tDofIndex = 0; tDofIndex < NumLocalDofsPerCell; tDofIndex++)
        {
          aOutput(tDofOffset + tDofIndex) = aInput(aCellOrdinal, tDofIndex);
        }
    }, "flatten residual vector");
}

/******************************************************************************//**
 * \brief Set all the entries in a 3-D matrix workset to a single value.
 *
 * \tparam NumRowsPerCell matrix number of rows
 * \tparam NumColumnsPerCell matrix number of columns
 * \tparam AViewType Output workset, as a 3-D Kokkos::View
 *
 * \param [in] aNumCells number of cells, i.e. elements
 * \param [in] aAlpha scalar multiplier
 * \param [in/out] aOutput 3-D matrix workset (NumCells, NumRowsPerCell, NumColumnsPerCell)
**********************************************************************************/
template<Plato::OrdinalType NumRowsPerCell, Plato::OrdinalType NumColumnsPerCell, class AViewType>
inline void fill_3D_workset(const Plato::OrdinalType& aNumCells,
                            typename AViewType::const_value_type & aAlpha,
                            AViewType& aOutput)
{
    if(aOutput.size() <= static_cast<Plato::OrdinalType>(0))
    {
        THROWERR("\nInput 3D array is empty, i.e. size <= 0.\n")
    }
    if(aNumCells <= static_cast<Plato::OrdinalType>(0))
    {
        THROWERR("\nInvalid number of input cells, i.e. elements. Value is <= 0.\n")
    }

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, aNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        for(Plato::OrdinalType tRowIndex = 0; tRowIndex < NumRowsPerCell; tRowIndex++)
        {
            for(Plato::OrdinalType tColIndex = 0; tColIndex < NumColumnsPerCell; tColIndex++)
            {
                aOutput(aCellOrdinal, tRowIndex, tColIndex) = aAlpha;
            }
        }
    }, "fill matrix identity 3DView");
}

/******************************************************************************//**
 * \brief Add two 3-D matrix workset
 *
 * \tparam AViewType Input matrix, as a 3-D Kokkos::View
 * \tparam BViewType Output matrix, as a 3-D Kokkos::View
 *
 * \param [in] aNumCells number of cells, i.e. elements
 * \param [in] aAlpha    scalar multiplier
 * \param [in] aA        3-D matrix workset (NumCells, NumRowsPerCell, NumColumnsPerCell)
 * \param [in] aBeta     scalar multiplier
 * \param [in/out] aB    3-D matrix workset (NumCells, NumRowsPerCell, NumColumnsPerCell)
**********************************************************************************/
template<class AViewType, class BViewType>
void update_matrix_workset(const Plato::OrdinalType& aNumCells,
                           typename AViewType::const_value_type& aAlpha,
                           const AViewType& aA,
                           typename BViewType::const_value_type& aBeta,
                           const BViewType& aB)
{
    if(aA.size() <= static_cast<Plato::OrdinalType>(0))
    {
        THROWERR("\nInput 3D array is empty, i.e. size <= 0\n")
    }
    if(aB.size() <= static_cast<Plato::OrdinalType>(0))
    {
        THROWERR("\nOutput 3D array is empty, i.e. size <= 0\n")
    }
    if(aA.extent(1) != aB.extent(1))
    {
        THROWERR("\nDimension mismatch, number of rows do not match.\n")
    }
    if(aA.extent(2) != aB.extent(2))
    {
        THROWERR("\nDimension mismatch, number of columns do not match.\n")
    }
    if(aNumCells <= static_cast<Plato::OrdinalType>(0))
    {
        THROWERR("\nNumber of input cells, i.e. elements, is less or equal to zero.\n");
    }

    const auto tNumRows = aA.extent(1);
    const auto tNumCols = aA.extent(2);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, aNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        for(Plato::OrdinalType tRowIndex = 0; tRowIndex < tNumRows; tRowIndex++)
        {
            for(Plato::OrdinalType tColIndex = 0; tColIndex < tNumCols; tColIndex++)
            {
                aB(aCellOrdinal, tRowIndex, tColIndex) = aAlpha * aA(aCellOrdinal, tRowIndex, tColIndex) +
                        aBeta * aB(aCellOrdinal, tRowIndex, tColIndex);
            }
        }
    }, "update matrix workset");
}

/******************************************************************************//**
 * \brief Add two 2-D vector workset
 *
 * \tparam XViewType Input matrix, as a 2-D Kokkos::View
 * \tparam YViewType Output matrix, as a 2-D Kokkos::View
 *
 * \param [in] aNumCells number of cells, i.e. elements
 * \param [in] aAlpha scalar multiplier
 * \param [in] aXvec 2-D vector workset (NumCells, NumEntriesPerCell)
 * \param [in] aBeta scalar multiplier
 * \param [in/out] aYvec 2-D vector workset (NumCells, NumEntriesPerCell)
**********************************************************************************/
template<class XViewType, class YViewType>
void update_vector_workset(const Plato::OrdinalType& aNumCells,
                           typename XViewType::const_value_type& aAlpha,
                           const XViewType& aXvec,
                           typename YViewType::const_value_type& aBeta,
                           const YViewType& aYvec)
{
    if(aXvec.size() != aYvec.size())
    {
        std::stringstream tMsg;
        tMsg << "\nDimension mismatch. Input vector size is " << aXvec.size()
                << " and output vector size is " << aYvec.size() << ".\n";
        THROWERR(tMsg.str().c_str())
    }
    if(aNumCells <= static_cast<Plato::OrdinalType>(0))
    {
        THROWERR("\nNumber of input cells, i.e. elements, is less or equal to zero.\n");
    }

    const auto tNumElements = aXvec.extent(1);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, aNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        for(Plato::OrdinalType tIndex = 0; tIndex < tNumElements; tIndex++)
        {
            aYvec(aCellOrdinal, tIndex) = aAlpha * aXvec(aCellOrdinal, tIndex) +
                    aBeta * aYvec(aCellOrdinal, tIndex);
        }
    }, "update vector workset");
}

/************************************************************************//**
 *
 * \brief Dense matrix-matrix multiplication: C = \f$ \beta*C + \alpha*op(A)*op(B)\f$.
 *        NOTE: Function does not support transpose operations
 *
 * \tparam AViewType Input matrix, as a 3-D Kokkos::View
 * \tparam BViewType Input matrix, as a 3-D Kokkos::View
 * \tparam CViewType Output matrix, as a nonconst 3-D Kokkos::View
 *
 * \param aNumCells [in]     Input number of cells, i.e. elements
 * \param aAlpha    [in]     Input coefficient of A
 * \param aA        [in]     Input matrix, as a 3-D Kokkos::View
 * \param aB        [in]     Input matrix, as a 3-D Kokkos::View
 * \param aBeta     [in]     Input coefficient of C
 * \param aC        [in/out] Output matrix, as a nonconst 3-D Kokkos::View
 *
****************************************************************************/
template<class AViewType, class BViewType, class CViewType>
void multiply_matrix_workset(const Plato::OrdinalType& aNumCells,
                             typename AViewType::const_value_type& aAlpha,
                             const AViewType& aA,
                             const BViewType& aB,
                             typename CViewType::const_value_type& aBeta,
                             CViewType& aC)
{
    if(aA.size() <= static_cast<Plato::OrdinalType>(0))
    {
        THROWERR("\nInput 3D array A is empty, i.e. size <= 0\n")
    }
    if(aB.size() <= static_cast<Plato::OrdinalType>(0))
    {
        THROWERR("\nInput 3D array B is empty, i.e. size <= 0\n")
    }
    if(aC.size() <= static_cast<Plato::OrdinalType>(0))
    {
        THROWERR("\nOutput 3D array C is empty, i.e. size <= 0\n")
    }
    if(aA.extent(1) != aB.extent(1))
    {
        THROWERR("\nDimension mismatch, input A and B matrices have different number of rows.\n")
    }
    if(aA.extent(2) != aB.extent(2))
    {
        THROWERR("\nDimension mismatch, input A and B matrices have different number of columns.\n")
    }
    if(aA.extent(1) != aC.extent(1))
    {
        THROWERR("\nDimension mismatch. Mismatch in input (A) and output (C) matrices row count.\n")
    }
    if(aA.extent(2) != aC.extent(2))
    {
        THROWERR("\nDimension mismatch. Mismatch in input (A) and output (C) matrices column count.\n")
    }
    if(aA.extent(0) != aNumCells)
    {
        THROWERR("\nDimension mismatch, number of cells of matrix A does not match input number of cells.\n")
    }
    if(aB.extent(0) != aNumCells)
    {
        THROWERR("\nDimension mismatch, number of cells of matrix B does not match input number of cells.\n")
    }
    if(aC.extent(0) != aNumCells)
    {
        THROWERR("\nDimension mismatch, number of cells of matrix C does not match input number of cells.\n")
    }

    const auto tNumRows = aA.extent(1);
    const auto tNumCols = aA.extent(2);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, aNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        for(Plato::OrdinalType tRowIndex = 0; tRowIndex < tNumRows; tRowIndex++)
        {
            for(Plato::OrdinalType tColIndex = 0; tColIndex < tNumCols; tColIndex++)
            {
                aC(aCellOrdinal, tRowIndex, tColIndex) = aBeta * aC(aCellOrdinal, tRowIndex, tColIndex);
            }
        }

        for(Plato::OrdinalType tOutRowIndex = 0; tOutRowIndex < tNumRows; tOutRowIndex++)
        {
            for(Plato::OrdinalType tCommonIndex = 0; tCommonIndex < tNumCols; tCommonIndex++)
            {
                for(Plato::OrdinalType tOutColIndex = 0; tOutColIndex < tNumCols; tOutColIndex++)
                {
                    aC(aCellOrdinal, tOutRowIndex, tOutColIndex) = aC(aCellOrdinal, tOutRowIndex, tOutColIndex) +
                            aAlpha * aA(aCellOrdinal, tOutRowIndex, tCommonIndex) * aB(aCellOrdinal, tCommonIndex, tOutColIndex);
                }
            }
        }
    }, "multiply matrix workset");
}

/************************************************************************//**
 *
 * \brief Dense matrix-vector multiply: y = beta*y + alpha*A*x.
 *
 * \tparam AViewType Input matrix, as a 2-D Kokkos::View
 * \tparam XViewType Input vector, as a 1-D Kokkos::View
 * \tparam YViewType Output vector, as a nonconst 1-D Kokkos::View
 * \tparam AlphaCoeffType Type of input coefficient alpha
 * \tparam BetaCoeffType Type of input coefficient beta
 *
 * \param trans [in] "N" for non-transpose, "T" for transpose.  All
 *   characters after the first are ignored.  This works just like
 *   the BLAS routines.
 * \param aAlpha [in]     Input coefficient of A*x
 * \param aAmat  [in]     Input matrix, as a 2-D Kokkos::View
 * \param aXvec  [in]     Input vector, as a 1-D Kokkos::View
 * \param aBeta  [in]     Input coefficient of y
 * \param aYvec  [in/out] Output vector, as a nonconst 1-D Kokkos::View
 *
********************************************************************************/
template<class AViewType, class XViewType, class YViewType>
void matrix_times_vector_workset(const char aTransA[],
                                 const Plato::OrdinalType& aNumCells,
                                 typename AViewType::const_value_type& aAlpha,
                                 const AViewType& aAmat,
                                 const XViewType& aXvec,
                                 typename YViewType::const_value_type& aBeta,
                                 YViewType& aYvec)
{
    // check validity of inputs' dimensions
    if(aAmat.size() <= static_cast<Plato::OrdinalType>(0))
    {
        THROWERR("\nInput matrix A is empty, i.e. size <= 0\n")
    }
    if(aXvec.size() <= static_cast<Plato::OrdinalType>(0))
    {
        THROWERR("\nInput vector X is empty, i.e. size <= 0\n")
    }
    if(aYvec.size() <= static_cast<Plato::OrdinalType>(0))
    {
        THROWERR("\nOutput vector Y is empty, i.e. size <= 0\n")
    }
    if(aAmat.extent(0) != aNumCells)
    {
        THROWERR("\nDimension mismatch, number of cells of matrix A does not match input number of cells.\n")
    }
    if(aXvec.extent(0) != aNumCells)
    {
        THROWERR("\nDimension mismatch, number of cells of input vector X does not match input number of cells.\n")
    }
    if(aYvec.extent(0) != aNumCells)
    {
        THROWERR("\nDimension mismatch, number of cells of output vector Y does not match input number of cells.\n")
    }

    // Check validity of transpose argument
    bool tValidTransA = (aTransA[0] == 'N') || (aTransA[0] == 'n') ||
                        (aTransA[0] == 'T') || (aTransA[0] == 't');

    if(!tValidTransA)
    {
        std::stringstream tMsg;
        tMsg << "\ntransA[0] = '" << aTransA[0] << "'. Valid values include 'N' or 'n' (No transpose) and 'T' or 't' (Transpose).\n";
        THROWERR(tMsg.str())
    }

    auto tNumRows = aAmat.extent(1);
    auto tNumCols = aAmat.extent(2);
    if((aTransA[0] == 'N') || (aTransA[0] == 'n'))
    {
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, aNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
        {
            for(Plato::OrdinalType tRowIndex = 0; tRowIndex < tNumRows; tRowIndex++)
            {
                aYvec(aCellOrdinal, tRowIndex) = aBeta * aYvec(aCellOrdinal, tRowIndex);
            }

            for(Plato::OrdinalType tRowIndex = 0; tRowIndex < tNumRows; tRowIndex++)
            {
                for(Plato::OrdinalType tColIndex = 0; tColIndex < tNumCols; tColIndex++)
                {
                    aYvec(aCellOrdinal, tRowIndex) = aYvec(aCellOrdinal, tRowIndex) +
                            aAlpha * aAmat(aCellOrdinal, tRowIndex, tColIndex) * aXvec(aCellOrdinal, tColIndex);
                }
            }
        }, "matrix vector multiplication - no transpose");
    }
    else
    {
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, aNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
        {
            for(Plato::OrdinalType tColIndex = 0; tColIndex < tNumCols; tColIndex++)
            {
                aYvec(aCellOrdinal, tColIndex) = aBeta * aYvec(aCellOrdinal, tColIndex);
            }

            for(Plato::OrdinalType tRowIndex = 0; tRowIndex < tNumRows; tRowIndex++)
            {
                for(Plato::OrdinalType tColIndex = 0; tColIndex < tNumCols; tColIndex++)
                {
                    aYvec(aCellOrdinal, tColIndex) = aYvec(aCellOrdinal, tColIndex) +
                            aAlpha * aAmat(aCellOrdinal, tRowIndex, tColIndex) * aXvec(aCellOrdinal, tRowIndex);
                }
            }
        }, "matrix vector multiplication - transpose");
    }
}

/************************************************************************//**
 *
 * \brief Convert 2-D view of automatic differentiation (AD) scalar types into
 *  3-D view of scalar types
 *
 * \tparam NumRowsPerCell number of rows per cell
 * \tparam NumColsPerCell number of columns per cell
 * \tparam ADType         AD scalar type
 *
 * \param aNumCells [in]     number of cells
 * \param aInput    [in]     2-D view of AD types
 * \param aOutput   [in/out] 3-D view of Scalar types
 *
********************************************************************************/
template<Plato::OrdinalType NumRowsPerCell, Plato::OrdinalType NumColsPerCell, typename ADType>
void convert_ad_type_to_scalar_type(const Plato::OrdinalType& aNumCells,
                                    const Plato::ScalarMultiVectorT<ADType>& aInput,
                                    Plato::ScalarArray3D& aOutput)
{
    if(aInput.size() <= static_cast<Plato::OrdinalType>(0))
    {
        THROWERR("\nInput 2D array size is zero.\n");
    }
    if(aNumCells <= static_cast<Plato::OrdinalType>(0))
    {
        THROWERR("\nNumber of input cells, i.e. elements, is zero.\n");
    }
    if(aOutput.size() <= static_cast<Plato::OrdinalType>(0))
    {
        THROWERR("\nOutput 3D array size is zero.\n");
    }

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, aNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
      for(Plato::OrdinalType tRowIndex = 0; tRowIndex < NumRowsPerCell; tRowIndex++)
      {
          for(Plato::OrdinalType tColumnIndex = 0; tColumnIndex < NumColsPerCell; tColumnIndex++)
          {
              aOutput(aCellOrdinal, tRowIndex, tColumnIndex) = aInput(aCellOrdinal, tRowIndex).dx(tColumnIndex);
          }
      }
    }, "convert AD type to Scalar type");
}

/************************************************************************//**
 *
 * \brief Build a workset of identity matrices
 *
 * \tparam NumRowsPerCell number of rows per cell
 * \tparam NumColsPerCell number of columns per cell
 *
 * \param aNumCells [in]     number of cells
 * \param aIdentity [in/out] 3-D view, workset of identity matrices
 *
********************************************************************************/
template<Plato::OrdinalType NumRowsPerCell, Plato::OrdinalType NumColumnsPerCell>
void identity_workset(const Plato::OrdinalType& aNumCells, Plato::ScalarArray3D& aIdentity)
{
    if(aIdentity.size() <= static_cast<Plato::OrdinalType>(0))
    {
        THROWERR("\nInput 3-D view is empty, i.e. size <= 0.\n")
    }
    if(aIdentity.extent(0) != aNumCells)
    {
        THROWERR("\nNumber of cell mismatch. Input array has different number of cells than input number of cell argument.\n")
    }

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, aNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        for(Plato::OrdinalType tRowIndex = 0; tRowIndex < NumRowsPerCell; tRowIndex++)
        {
            for(Plato::OrdinalType tColumnIndex = 0; tColumnIndex < NumColumnsPerCell; tColumnIndex++)
            {
                aIdentity(aCellOrdinal, tRowIndex, tColumnIndex) = tRowIndex == tColumnIndex ? 1.0 : 0.0;
            }
        }
    }, "identity workset");
}

/************************************************************************//**
 *
 * \brief Compute the inverse of each matrix in workset
 *
 * \tparam NumRowsPerCell number of rows per cell
 * \tparam NumColsPerCell number of columns per cell
 * \tparam AViewType      Input matrix, as a 3-D Kokkos::View
 * \tparam BViewType      Output matrix, as a 3-D Kokkos::View
 *
 * \param aNumCells [in]     number of cells
 * \param aA        [in]     3-D view, matrix workset
 * \param aInverse  [in/out] 3-D view, matrix inverse workset
 *
********************************************************************************/
template<Plato::OrdinalType NumRowsPerCell, Plato::OrdinalType NumColumnsPerCell, class AViewType, class BViewType>
void inverse_matrix_workset(const Plato::OrdinalType& aNumCells, AViewType& aA, BViewType& aInverse)
{
    if(aA.size() <= static_cast<Plato::OrdinalType>(0))
    {
        THROWERR("\nInput 3D array, i.e. matrix workset, size is zero.\n")
    }
    if(aInverse.size() <= static_cast<Plato::OrdinalType>(0))
    {
        THROWERR("\nOutput 3D array, i.e. matrix workset, size is zero.\n")
    }
    if(aA.size() != aInverse.size())
    {
        THROWERR("\nInput and output views dimensions are different, i.e. Input.size != Output.size.\n")
    }

    Plato::identity_workset<NumRowsPerCell, NumColumnsPerCell>(aNumCells, aInverse);

    using namespace KokkosBatched::Experimental;
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, aNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        auto tA = Kokkos::subview(aA, aCellOrdinal, Kokkos::ALL(), Kokkos::ALL());
        auto tAinv = Kokkos::subview(aInverse, aCellOrdinal, Kokkos::ALL(), Kokkos::ALL());

        const Plato::Scalar tAlpha = 1.0;
        SerialLU<Algo::LU::Blocked>::invoke(tA);
        SerialTrsm<Side::Left,Uplo::Lower,Trans::NoTranspose,Diag::Unit   ,Algo::Trsm::Blocked>::invoke(tAlpha, tA, tAinv);
        SerialTrsm<Side::Left,Uplo::Upper,Trans::NoTranspose,Diag::NonUnit,Algo::Trsm::Blocked>::invoke(tAlpha, tA, tAinv);
    }, "compute matrix inverse 3DView");
}

/***************************************************************************//**
 *
 * \brief Abstract vector function interface for Variational Multi-Scale (VMS)
 *   Partial Differential Equations (PDEs) with history dependent states
 * \tparam EvaluationType evaluation type use to determine automatic differentiation
 *   type for the vector function (e.g. Residual, Jacobian, GradientZ, GradientU, etc.)
 *
*******************************************************************************/
template<typename EvaluationType>
class AbstractVectorFunctionVMSInc
{
// Protected member data
protected:
    Omega_h::Mesh &mMesh; /*!< mesh database */
    Plato::DataMap &mDataMap; /*!< output database */
    Omega_h::MeshSets &mMeshSets; /*!< sideset database */

// Public access functions
public:
    /***************************************************************************//**
     * \brief Constructor
     * \param [in]  aMesh mesh metadata
     * \param [in]  aMeshSets mesh side-sets metadata
     * \param [in]  aDataMap output data map
    *******************************************************************************/
    explicit AbstractVectorFunctionVMSInc(Omega_h::Mesh &aMesh,
                                               Omega_h::MeshSets &aMeshSets,
                                               Plato::DataMap &aDataMap) :
        mMesh(aMesh),
        mDataMap(aDataMap),
        mMeshSets(aMeshSets)
    {
    }

    /***************************************************************************//**
     * \brief Destructor
    *******************************************************************************/
    virtual ~AbstractVectorFunctionVMSInc()
    {
    }

    /***************************************************************************//**
     * \brief Return reference to Omega_h mesh data base
     * \return mesh metadata
    *******************************************************************************/
    decltype(mMesh) getMesh() const
    {
        return (mMesh);
    }

    /***************************************************************************//**
     * \brief Return reference to Omega_h mesh sets
     * \return mesh side sets metadata
    *******************************************************************************/
    decltype(mMeshSets) getMeshSets() const
    {
        return (mMeshSets);
    }

    /***************************************************************************//**
     *
     * \brief Evaluate the stabilized residual equation
     *
     * \param [in] aGlobalState current global state ( i.e. state at the n-th time interval (\f$ t^{n} \f$) )
     * \param [in] aGlobalStatePrev previous global state ( i.e. state at the n-th minus one time interval (\f$ t^{n-1} \f$) )
     * \param [in] aLocalState current local state ( i.e. state at the n-th time interval (\f$ t^{n} \f$) )
     * \param [in] aLocalStatePrev previous local state ( i.e. state at the n-th minus one time interval (\f$ t^{n-1} \f$) )
     * \param [in] aPressureGrad current pressure gradient ( i.e. state at the n-th time interval (\f$ t^{n} \f$) )
     * \param [in] aControl design variables
     * \param [in/out] aResult residual evaluation
     * \param [in] aTimeStep current time step (i.e. \f$ \Delta{t}^{n} \f$), default = 0.0
     *
    *******************************************************************************/
    virtual void
    evaluate(const Plato::ScalarMultiVectorT<typename EvaluationType::StateScalarType> &aGlobalState,
             const Plato::ScalarMultiVectorT<typename EvaluationType::PrevStateScalarType> &aGlobalStatePrev,
             const Plato::ScalarMultiVectorT<typename EvaluationType::LocalStateScalarType> &aLocalState,
             const Plato::ScalarMultiVectorT<typename EvaluationType::PrevLocalStateScalarType> &aLocalStatePrev,
             const Plato::ScalarMultiVectorT<typename EvaluationType::NodeStateScalarType> &aPressureGrad,
             const Plato::ScalarMultiVectorT<typename EvaluationType::ControlScalarType> &aControl,
             const Plato::ScalarArray3DT<typename EvaluationType::ConfigScalarType> &aConfig,
             const Plato::ScalarMultiVectorT<typename EvaluationType::ResultScalarType> &aResult,
             Plato::Scalar aTimeStep = 0.0) = 0;
};
// class AbstractVectorFunctionVMSInc






/***************************************************************************//**
 *
 * \brief Apply penalty, i.e. density penalty, to 2-D view
 *
 * \tparam Length      number of data entries for a given cell
 * \tparam ControlType penalty, as a Scalar
 * \tparam ResultType  multi-vector, as a 3-D Kokkos::View
 *
 * \param [in] aCellOrdinal cell ordinal, i.e. index
 * \param [in] aPenalty     material penalty
 * \param [in] aOutput      physical quantity to be penalized
 *
*******************************************************************************/
template<Plato::OrdinalType Length, typename ControlType, typename ResultType>
DEVICE_TYPE inline void
apply_penalty(const Plato::OrdinalType aCellOrdinal, const ControlType & aPenalty, const Plato::ScalarMultiVectorT<ResultType> & aOutput)
{
    for(Plato::OrdinalType tIndex = 0; tIndex < Length; tIndex++)
    {
        aOutput(aCellOrdinal, tIndex) *= aPenalty;
    }
}

/***************************************************************************//**
 *
 * \brief Compute shear modulus
 *
 * \tparam ScalarType POD type
 *
 * \param [in] aElasticModulus elastic modulus
 * \param [in] aPoissonRatio   poisson's ratio
 * \return shear modulus
 *
*******************************************************************************/
template<typename ScalarType>
inline ScalarType compute_shear_modulus(const ScalarType & aElasticModulus, const ScalarType & aPoissonRatio)
{
    ScalarType tShearModulus = aElasticModulus /
        ( static_cast<Plato::Scalar>(2) * ( static_cast<Plato::Scalar>(1) + aPoissonRatio) ) ;
    return (tShearModulus);
}

/***************************************************************************//**
 *
 * \brief Compute bulk modulus
 *
 * \tparam ScalarType POD type
 *
 * \param [in] aElasticModulus elastic modulus
 * \param [in] aPoissonRatio   poisson's ratio
 * \return bulk modulus
 *
*******************************************************************************/
template<typename ScalarType>
inline ScalarType compute_bulk_modulus(const ScalarType & aElasticModulus, const ScalarType & aPoissonRatio)
{
    ScalarType tShearModulus = aElasticModulus /
        ( static_cast<Plato::Scalar>(3) * ( static_cast<Plato::Scalar>(1) - ( static_cast<Plato::Scalar>(2) * aPoissonRatio) ) );
    return (tShearModulus);
}








Plato::Scalar parse_elastic_modulus(Teuchos::ParameterList & aParamList)
{
    if(aParamList.isParameter("Youngs Modulus"))
    {
        Plato::Scalar tElasticModulus = aParamList.get<Plato::Scalar>("Youngs Modulus");
        return (tElasticModulus);
    }
    else
    {
        THROWERR("Youngs Modulus parameter is not defined in 'Isotropic Linear Elastic' sublist.")
    }
}

Plato::Scalar parse_poissons_ratio(Teuchos::ParameterList & aParamList)
{
    if(aParamList.isParameter("Poissons Ratio"))
    {
        Plato::Scalar tPoissonsRatio = aParamList.get<Plato::Scalar>("Poissons Ratio");
        return (tPoissonsRatio);
    }
    else
    {
        THROWERR("Poisson's ratio parameter is not defined in 'Isotropic Linear Elastic' sublist.")
    }
}





/***************************************************************************//**
 *
 * \brief Apply the divergence operator to the strain tensor, i.e.
 *   /f$ \div\cdot\epsilon /f$, where /f$ \epsilon /f$ denotes the strain tensor.
 *
 * \tparam SpaceDim spatial dimensions
 *
*******************************************************************************/
template<Plato::OrdinalType SpaceDim>
class StrainDivergence
{
public:
    /***************************************************************************//**
     * \brief Constructor
    *******************************************************************************/
    StrainDivergence(){}

    /***************************************************************************//**
     * \brief Destructor
    *******************************************************************************/
    ~StrainDivergence(){}

    /***************************************************************************//**
     *
     * \brief Apply the divergence operator to the strain tensor.
     *
     * \tparam StrainType POD type for 2-D Kokkos::View
     * \tparam ResultType POD type for 1-D Kokkos::View
     *
     * \param [in] aCellOrdinal cell ordinal, i.e. index
     * \param [in] aStrain      strain tensor
     * \param [in] aOutput      strain tensor divergence
     *
    *******************************************************************************/
    template<typename StrainType, typename ResultType>
    DEVICE_TYPE inline void
    operator()(const Plato::OrdinalType & aCellOrdinal,
               const Plato::ScalarMultiVectorT<StrainType> & aStrain,
               const Plato::ScalarVectorT<ResultType> & aOutput) const;
};
// class StrainDivergence

/***************************************************************************//**
 *
 * \brief Specialization for 3-D applications.
 *
 * \tparam StrainType POD type for 2-D Kokkos::View
 * \tparam ResultType POD type for 1-D Kokkos::View
 *
 * \param [in] aCellOrdinal cell ordinal, i.e. index
 * \param [in] aStrain      strain tensor
 * \param [in] aOutput      strain tensor divergence
 *
*******************************************************************************/
template<>
template<typename StrainType, typename ResultType>
DEVICE_TYPE inline void
StrainDivergence <3>::operator()(const Plato::OrdinalType & aCellOrdinal,
                                 const Plato::ScalarMultiVectorT<StrainType> & aStrain,
                                 const Plato::ScalarVectorT<ResultType> & aOutput) const
{
    aOutput(aCellOrdinal) = aStrain(aCellOrdinal, 0) + aStrain(aCellOrdinal, 1) + aStrain(aCellOrdinal, 2);
}

/***************************************************************************//**
 *
 * \brief Specialization for 2-D applications.
 *
 * \tparam StrainType POD type for 2-D Kokkos::View
 * \tparam ResultType POD type for 1-D Kokkos::View
 *
 * \param [in] aCellOrdinal cell ordinal, i.e. index
 * \param [in] aStrain      strain tensor
 * \param [in] aOutput      strain tensor divergence
 *
*******************************************************************************/
template<>
template<typename StrainType, typename ResultType>
DEVICE_TYPE inline void
StrainDivergence <2>::operator()(const Plato::OrdinalType & aCellOrdinal,
                                 const Plato::ScalarMultiVectorT<StrainType> & aStrain,
                                 const Plato::ScalarVectorT<ResultType> & aOutput) const
{
    aOutput(aCellOrdinal) = aStrain(aCellOrdinal, 0) + aStrain(aCellOrdinal, 1);
}

/***************************************************************************//**
 *
 * \brief Specialization for 1-D applications.
 *
 * \tparam StrainType POD type for 2-D Kokkos::View
 * \tparam ResultType POD type for 1-D Kokkos::View
 *
 * \param [in] aCellOrdinal cell ordinal, i.e. index
 * \param [in] aStrain      strain tensor
 * \param [in] aOutput      strain tensor divergence
 *
*******************************************************************************/
template<>
template<typename StrainType, typename ResultType>
DEVICE_TYPE inline void
StrainDivergence <1>::operator()(const Plato::OrdinalType & aCellOrdinal,
                                 const Plato::ScalarMultiVectorT<StrainType> & aStrain,
                                 const Plato::ScalarVectorT<ResultType> & aOutput) const
{
    aOutput(aCellOrdinal) = aStrain(aCellOrdinal, 0);
}






/***************************************************************************//**
 *
 * \brief Compute stabilization term, which is given by:
 *
 *   /f$ \tau\nabla{p} - \nabla{\Pi} /f$, where /f$\tau/f$ is the stabilization
 *   multiplier, /f$ \nabla{p} /f$ is the pressure gradient, and /f$ \nabla{\Pi} /f$
 *   is the projected pressure gradient.  The stabilization parameter, /f$ \tau /f$
 *   is defined as /f$ \frac{\Omega_{e}^{2/3}}{2G} /f$, where G is the shear modulus
 *
 * \tparam SpaceDim spatial dimensions
 *
*******************************************************************************/
template<Plato::OrdinalType SpaceDim>
class ComputeStabilization
{
private:
    Plato::Scalar mTwoOverThree;         /*!< 2/3 constant - avoids repeated calculation */
    Plato::Scalar mPressureScaling;      /*!< pressure scaling term */
    Plato::Scalar mElasticShearModulus;  /*!< elastic shear modulus */

public:
    /***************************************************************************//**
     * \brief Constructor
     * \param [in] aScaling       multiplier used to improve the system of equations condition number
     * \param [in] aShearModulus  elastic shear modulus
    *******************************************************************************/
    explicit ComputeStabilization(const Plato::Scalar & aScaling, const Plato::Scalar & aShearModulus) :
        mTwoOverThree(2.0/3.0),
        mPressureScaling(aScaling),
        mElasticShearModulus(aShearModulus)
    {
    }

    /***************************************************************************//**
     * \brief Destructor
    *******************************************************************************/
    ~ComputeStabilization(){}

    /***************************************************************************//**
     * \brief Compute stabilization term
     *
     * \tparam ConfigT        POD type for 1-D Kokkos::View
     * \tparam PressGradT     POD type for 2-D Kokkos::View
     * \tparam ProjPressGradT POD type for 2-D Kokkos::View
     * \tparam ResultT        POD type for 2-D Kokkos::View
     *
     * \param [in] aCellOrdinal           cell ordinal, i.e index
     * \param [in] aCellVolume            cell volume
     * \param [in] aPressureGrad          pressure gradient
     * \param [in] aProjectedPressureGrad projected pressure gradient
     * \param [in/out] aStabilization     stabilization term
     *
    *******************************************************************************/
    template<typename ConfigT, typename PressGradT, typename ProjPressGradT, typename ResultT>
    DEVICE_TYPE inline void
    operator()(const Plato::OrdinalType &aCellOrdinal,
               const Plato::ScalarVectorT<ConfigT> & aCellVolume,
               const Plato::ScalarMultiVectorT<PressGradT> &aPressureGrad,
               const Plato::ScalarMultiVectorT<ProjPressGradT> &aProjectedPressureGrad,
               const Plato::ScalarMultiVectorT<ResultT> &aStabilization) const;
};
// class ComputeStabilization

/***************************************************************************//**
 *
 * \brief Specialization for 3-D applications
 *
 * \tparam ConfigT        POD type for 1-D Kokkos::View
 * \tparam PressGradT     POD type for 2-D Kokkos::View
 * \tparam ProjPressGradT POD type for 2-D Kokkos::View
 * \tparam ResultT        POD type for 2-D Kokkos::View
 *
 * \param [in] aCellOrdinal           cell ordinal, i.e index
 * \param [in] aCellVolume            cell volume
 * \param [in] aPressureGrad          pressure gradient
 * \param [in] aProjectedPressureGrad projected pressure gradient
 * \param [in/out] aStabilization     stabilization term
 *
*******************************************************************************/
template<>
template<typename ConfigT, typename PressGradT, typename ProjPressGradT, typename ResultT>
DEVICE_TYPE inline void
ComputeStabilization<3>::operator()(const Plato::OrdinalType & aCellOrdinal,
                                    const Plato::ScalarVectorT<ConfigT> & aCellVolume,
                                    const Plato::ScalarMultiVectorT<PressGradT> & aPressureGrad,
                                    const Plato::ScalarMultiVectorT<ProjPressGradT> & aProjectedPressureGrad,
                                    const Plato::ScalarMultiVectorT<ResultT> & aStabilization) const
{
    ConfigT tTau = pow(aCellVolume(aCellOrdinal), mTwoOverThree) / (static_cast<Plato::Scalar>(2.0) * mElasticShearModulus);
    aStabilization(aCellOrdinal, 0) = mPressureScaling * tTau
        * ((mPressureScaling * aPressureGrad(aCellOrdinal, 0)) - aProjectedPressureGrad(aCellOrdinal, 0));

    aStabilization(aCellOrdinal, 1) = mPressureScaling * tTau
        * ((mPressureScaling * aPressureGrad(aCellOrdinal, 1)) - aProjectedPressureGrad(aCellOrdinal, 1));

    aStabilization(aCellOrdinal, 2) = mPressureScaling * tTau
        * ((mPressureScaling * aPressureGrad(aCellOrdinal, 2)) - aProjectedPressureGrad(aCellOrdinal, 2));
}

/***************************************************************************//**
 *
 * \brief Specialization for 2-D applications
 *
 * \tparam ConfigT        POD type for 1-D Kokkos::View
 * \tparam PressGradT     POD type for 2-D Kokkos::View
 * \tparam ProjPressGradT POD type for 2-D Kokkos::View
 * \tparam ResultT        POD type for 2-D Kokkos::View
 *
 * \param [in] aCellOrdinal           cell ordinal, i.e index
 * \param [in] aCellVolume            cell volume
 * \param [in] aPressureGrad          pressure gradient
 * \param [in] aProjectedPressureGrad projected pressure gradient
 * \param [in/out] aStabilization     stabilization term
 *
*******************************************************************************/
template<>
template<typename ConfigT, typename PressGradT, typename ProjPressGradT, typename ResultT>
DEVICE_TYPE inline void
ComputeStabilization<2>::operator()(const Plato::OrdinalType & aCellOrdinal,
                                    const Plato::ScalarVectorT<ConfigT> & aCellVolume,
                                    const Plato::ScalarMultiVectorT<PressGradT> & aPressureGrad,
                                    const Plato::ScalarMultiVectorT<ProjPressGradT> & aProjectedPressureGrad,
                                    const Plato::ScalarMultiVectorT<ResultT> & aStabilization) const
{
    ConfigT tTau = pow(aCellVolume(aCellOrdinal), mTwoOverThree) / (static_cast<Plato::Scalar>(2.0) * mElasticShearModulus);
    aStabilization(aCellOrdinal, 0) = mPressureScaling * tTau
        * (mPressureScaling * aPressureGrad(aCellOrdinal, 0) - aProjectedPressureGrad(aCellOrdinal, 0));

    aStabilization(aCellOrdinal, 1) = mPressureScaling * tTau
        * (mPressureScaling * aPressureGrad(aCellOrdinal, 1) - aProjectedPressureGrad(aCellOrdinal, 1));
}

/***************************************************************************//**
 *
 * \brief Specialization for 1-D applications
 *
 * \tparam ConfigT        POD type for 1-D Kokkos::View
 * \tparam PressGradT     POD type for 2-D Kokkos::View
 * \tparam ProjPressGradT POD type for 2-D Kokkos::View
 * \tparam ResultT        POD type for 2-D Kokkos::View
 *
 * \param [in] aCellOrdinal           cell ordinal, i.e index
 * \param [in] aCellVolume            cell volume
 * \param [in] aPressureGrad          pressure gradient
 * \param [in] aProjectedPressureGrad projected pressure gradient
 * \param [in/out] aStabilization     stabilization term
 *
*******************************************************************************/
template<>
template<typename ConfigT, typename PressGradT, typename ProjPressGradT, typename ResultT>
DEVICE_TYPE inline void
ComputeStabilization<1>::operator()(const Plato::OrdinalType & aCellOrdinal,
                                    const Plato::ScalarVectorT<ConfigT> & aCellVolume,
                                    const Plato::ScalarMultiVectorT<PressGradT> & aPressureGrad,
                                    const Plato::ScalarMultiVectorT<ProjPressGradT> & aProjectedPressureGrad,
                                    const Plato::ScalarMultiVectorT<ResultT> & aStabilization) const
{
    ConfigT tTau = pow(aCellVolume(aCellOrdinal), mTwoOverThree) / (static_cast<Plato::Scalar>(2.0) * mElasticShearModulus);
    aStabilization(aCellOrdinal, 0) = mPressureScaling * tTau
        * (mPressureScaling * aPressureGrad(aCellOrdinal, 0) - aProjectedPressureGrad(aCellOrdinal, 0));
}








/***********************************************************************//**
 * \brief Evaluate stabilized elasto-plastic residual, defined as
 *
 * \f$   \langle \nabla{v_h}, s_h \rangle + \langle \nabla\cdot{v_h}, p_h \rangle
 *     - \langle v_h, f \rangle - \langle v_h, b \rangle = 0\ \forall\ \v_h \in V_{h,0}
 *       = \{v_h \in V_h | v = 0\ \mbox{in}\ \partial\Omega_{u} \} \f$
 *
 * \f$   \langle q_h, \nabla\cdot{u_h} \rangle - \langle q_h, \frac{1}{K}p_h \rangle
 *     - \sum_{e=1}^{N_{elem}} \tau_e \langle \nabla{q_h} \left[ \nabla{p_h} - \Pi_h
 *       \right] \rangle = 0\ \forall\ q_h \in L_h \subset\ L^2(\Omega) \f$
 *
 * \f$ \langle \nabla{p_h}, \eta_h \rangle - \langle \Pi_h, \eta_h \rangle = 0\
 *     \forall\ \eta_h \in V_h \subset\ H^{1}(\Omega) \f$
 *
***************************************************************************/
template<typename EvaluationType, typename PhysicsType>
class ElastoPlasticityResidual: public Plato::AbstractVectorFunctionVMSInc<EvaluationType>
{
// Private member data
private:
    static constexpr auto mSpaceDim = EvaluationType::SpatialDim;               /*!< number of spatial dimensions */
    static constexpr auto mNumVoigtTerms = PhysicsType::mNumVoigtTerms;         /*!< number of voigt terms */
    static constexpr auto mNumDofsPerCell = PhysicsType::mNumDofsPerCell;       /*!< number of degrees of freedom (dofs) per cell */
    static constexpr auto mNumDofsPerNode = PhysicsType::mNumDofsPerNode;       /*!< number of dofs per node */
    static constexpr auto mNumNodesPerCell = PhysicsType::mNumNodesPerCell;     /*!< number nodes per cell */
    static constexpr auto mPressureDofOffset = PhysicsType::mPressureDofOffset; /*!< number of pressure dofs offset */

    static constexpr auto mNumMechDims = mSpaceDim;         /*!< number of mechanical degrees of freedom */
    static constexpr Plato::OrdinalType mMechDofOffset = 0; /*!< mechanical degrees of freedom offset */

    using Plato::AbstractVectorFunctionVMSInc<EvaluationType>::mMesh;     /*!< mesh database */
    using Plato::AbstractVectorFunctionVMSInc<EvaluationType>::mDataMap;  /*!< PLATO Engine output database */
    using Plato::AbstractVectorFunctionVMSInc<EvaluationType>::mMeshSets; /*!< side-sets metadata */

    using GlobalStateT = typename EvaluationType::StateScalarType;             /*!< global state variables automatic differentiation type */
    using PrevGlobalStateT = typename EvaluationType::PrevStateScalarType;     /*!< global state variables automatic differentiation type */
    using LocalStateT = typename EvaluationType::LocalStateScalarType;         /*!< local state variables automatic differentiation type */
    using PrevLocalStateT = typename EvaluationType::PrevLocalStateScalarType; /*!< local state variables automatic differentiation type */
    using NodeStateT = typename EvaluationType::NodeStateScalarType;           /*!< node State AD type */
    using ControlT = typename EvaluationType::ControlScalarType;               /*!< control variables automatic differentiation type */
    using ConfigT = typename EvaluationType::ConfigScalarType;                 /*!< config variables automatic differentiation type */
    using ResultT = typename EvaluationType::ResultScalarType;                 /*!< result variables automatic differentiation type */

    Plato::Scalar mPoissonsRatio;                  /*!< Poisson's ratio */
    Plato::Scalar mElasticModulus;                 /*!< elastic modulus */
    Plato::Scalar mPressureScaling;                /*!< Pressure scaling term */
    Plato::Scalar mElasticBulkModulus;             /*!< elastic bulk modulus */
    Plato::Scalar mElasticShearModulus;            /*!< elastic shear modulus */
    Plato::Scalar mElasticPropertiesPenaltySIMP;   /*!< SIMP penalty for elastic properties */
    Plato::Scalar mElasticPropertiesMinErsatzSIMP; /*!< SIMP min ersatz stiffness for elastic properties */

    std::vector<std::string> mPlotTable; /*!< array with output data identifiers*/

    std::shared_ptr<Plato::BodyLoads<EvaluationType>> mBodyLoads;               /*!< body loads interface */
    std::shared_ptr<Plato::LinearTetCubRuleDegreeOne<mSpaceDim>> mCubatureRule; /*!< linear cubature rule */

// Private access functions
private:
    /***************************************************************************//**
     * \brief initialize material, loads and output data
     * \param [in] aProblemParams input XML data, i.e. parameter list
    *******************************************************************************/
    void initialize(Teuchos::ParameterList &aProblemParams)
    {
        this->parseExternalForces(aProblemParams);
        this->parseOutputDataNames(aProblemParams);
        this->parseMaterialProperties(aProblemParams);
        this->parseMaterialPenaltyInputs(aProblemParams);
    }

    /***************************************************************************//**
     * \brief Parse output data names
     * \param [in] aProblemParams input XML data, i.e. parameter list
    *******************************************************************************/
    void parseOutputDataNames(Teuchos::ParameterList &aProblemParams)
    {
        if(aProblemParams.isSublist("Elliptic"))
        {
            auto tResidualParams = aProblemParams.sublist("Elliptic");
            if (tResidualParams.isType < Teuchos::Array < std::string >> ("Plottable"))
            {
                mPlotTable = tResidualParams.get < Teuchos::Array < std::string >> ("Plottable").toVector();
            }
        }
    }

    /***************************************************************************//**
     * \brief Parse material penalty inputs
     * \param [in] aProblemParams input XML data, i.e. parameter list
    *******************************************************************************/
    void parseMaterialPenaltyInputs(Teuchos::ParameterList &aProblemParams)
    {
        if(aProblemParams.isSublist("Elliptic"))
        {
            auto tResidualParams = aProblemParams.sublist("Elliptic");
            if(tResidualParams.isSublist("Penalty Function"))
            {
                auto tPenaltyParams = tResidualParams.sublist("Penalty Function");
                mElasticPropertiesPenaltySIMP = tPenaltyParams.get<Plato::Scalar>("Exponent", 3.0);
                mElasticPropertiesMinErsatzSIMP = tPenaltyParams.get<Plato::Scalar>("Minimum Value", 1e-9);
            }
        }
    }

    /***********************************************************************//**
     * \brief Parse external forces
     * \param [in] aProblemParams input XML data, i.e. parameter list
    ***************************************************************************/
    void parseExternalForces(Teuchos::ParameterList &aProblemParams)
    {
        // Parse body loads
        if (aProblemParams.isSublist("Body Loads"))
        {
            mBodyLoads = std::make_shared<Plato::BodyLoads<EvaluationType>>(aProblemParams.sublist("Body Loads"));
        }
    }

    /**********************************************************************//**
     * \brief Parse elastic material properties
     * \param [in] aProblemParams input XML data, i.e. parameter list
    **************************************************************************/
    void parseMaterialProperties(Teuchos::ParameterList &aProblemParams)
    {
        if(aProblemParams.isSublist("Material Model"))
        {
            this->parseIsotropicMaterialProperties(aProblemParams);
        }
        else
        {
            THROWERR("'Material Model' sublist is not defined.")
        }
    }

    /**********************************************************************//**
     * \brief Parse isotropic material parameters
     * \param [in] aProblemParams input XML data, i.e. parameter list
    **************************************************************************/
    void parseIsotropicMaterialProperties(Teuchos::ParameterList &aProblemParams)
    {
        auto tMaterialInputs = aProblemParams.get<Teuchos::ParameterList>("Material Model");
        if (tMaterialInputs.isSublist("Isotropic Linear Elastic"))
        {
            auto tElasticSubList = tMaterialInputs.sublist("Isotropic Linear Elastic");
            mPoissonsRatio = Plato::parse_poissons_ratio(tElasticSubList);
            mElasticModulus = Plato::parse_elastic_modulus(tElasticSubList);
            mElasticBulkModulus = Plato::compute_bulk_modulus(mElasticModulus, mPoissonsRatio);
            mElasticShearModulus = Plato::compute_shear_modulus(mElasticModulus, mPoissonsRatio);
            this->parsePressureTermScaling(tMaterialInputs);
        }
        else
        {
            THROWERR("'Isotropic Linear Elastic' sublist of 'Material Model' is not defined.")
        }
    }

    /**********************************************************************//**
     * \brief Parse pressure scaling, needed to improve the condition number.
     * \param [in] aMatParamList material model inputs, i.e. parameter list
    **************************************************************************/
    void parsePressureTermScaling(Teuchos::ParameterList & aMatParamList)
    {
        if (aMatParamList.isType<Plato::Scalar>("Pressure Scaling"))
        {
            mPressureScaling = aMatParamList.get<Plato::Scalar>("Pressure Scaling");
        }
        else
        {
            mPressureScaling = mElasticBulkModulus;
        }
    }

    /**********************************************************************//**
     * \brief Copy data to output data map
     * \tparam DataT data type
     * \param [in] aData output data
     * \param [in] aName output data name
    **************************************************************************/
    template<typename DataT>
    void outputData(const DataT & aData, const std::string & aName)
    {
        if(std::count(mPlotTable.begin(), mPlotTable.end(), aName))
        {
            toMap(mDataMap, aData, aName);
        }
    }

    /************************************************************************//**
     * \brief Add external forces to residual
     * \param [in]     aGlobalState current global state ( i.e. state at the n-th time interval (\f$ t^{n} \f$) )
     * \param [in]     aControl     design variables
     * \param [in/out] aResult      residual evaluation
    ****************************************************************************/
    void addExternalForces(const Plato::ScalarMultiVectorT<GlobalStateT> &aGlobalState,
                           const Plato::ScalarMultiVectorT<ControlT> &aControl,
                           const Plato::ScalarMultiVectorT<ResultT> &aResult)
    {
        if (mBodyLoads != nullptr)
        {
            Plato::Scalar tMultiplier = -1.0;
            mBodyLoads->get(mMesh, aGlobalState, aControl, aResult, tMultiplier);
        }
    }

// Public access functions
public:
    /***************************************************************************//**
     * \brief Constructor
     * \param [in] aMesh          mesh metadata
     * \param [in] aMeshSets      side-sets metadata
     * \param [in] aDataMap       output data map
     * \param [in] aProblemParams input XML data
     * \param [in] aPenaltyParams penalty function input XML data
    *******************************************************************************/
    ElastoPlasticityResidual(Omega_h::Mesh &aMesh,
                             Omega_h::MeshSets &aMeshSets,
                             Plato::DataMap &aDataMap,
                             Teuchos::ParameterList &aProblemParams) :
        Plato::AbstractVectorFunctionVMSInc<EvaluationType>(aMesh, aMeshSets, aDataMap),
        mPoissonsRatio(-1.0),
        mElasticModulus(-1.0),
        mPressureScaling(1.0),
        mElasticBulkModulus(-1.0),
        mElasticShearModulus(-1.0),
        mElasticPropertiesPenaltySIMP(3),
        mElasticPropertiesMinErsatzSIMP(1e-9),
        mBodyLoads(nullptr),
        mCubatureRule(std::make_shared<Plato::LinearTetCubRuleDegreeOne<mSpaceDim>>())
    {
        this->initialize(aProblemParams);
    }

    /***************************************************************************//**
     * \brief Destructor
    *******************************************************************************/
    virtual ~ElastoPlasticityResidual()
    {
    }

    /************************************************************************//**
     * \brief Evaluate the stabilized residual equation
     *
     * \param [in]     aCurrentGlobalState    current global state workset ( i.e. state at the n-th time interval (\f$ t^{n} \f$) )
     * \param [in]     aPrevGlobalState       previous global state workset ( i.e. state at the n-th minus one time interval (\f$ t^{n-1} \f$) )
     * \param [in]     aCurrentLocalState     current local state workset ( i.e. state at the n-th time interval (\f$ t^{n} \f$) )
     * \param [in]     aPrevLocalState        previous local state workset ( i.e. state at the n-th minus one time interval (\f$ t^{n-1} \f$) )
     * \param [in]     aProjectedPressureGrad current pressure gradient workset ( i.e. state at the n-th time interval (\f$ t^{n} \f$) )
     * \param [in]     aControl               design variables workset
     * \param [in]     aConfig                configuration workset
     * \param [in/out] aResult                residual workset
     * \param [in]     aTimeStep              current time step (i.e. \f$ \Delta{t}^{n} \f$), default = 0.0
     *
    ****************************************************************************/
    void evaluate(const Plato::ScalarMultiVectorT<GlobalStateT> &aCurrentGlobalState,
                  const Plato::ScalarMultiVectorT<PrevGlobalStateT> &aPrevGlobalState,
                  const Plato::ScalarMultiVectorT<LocalStateT> &aCurrentLocalState,
                  const Plato::ScalarMultiVectorT<PrevLocalStateT> &aPrevLocalState,
                  const Plato::ScalarMultiVectorT<NodeStateT> &aProjectedPressureGrad,
                  const Plato::ScalarMultiVectorT<ControlT> &aControl,
                  const Plato::ScalarArray3DT<ConfigT> &aConfig,
                  const Plato::ScalarMultiVectorT<ResultT> &aResult,
                  Plato::Scalar aTimeStep = 0.0) override
    {
        auto tNumCells = mMesh.nelems();

        using GradScalarT = typename Plato::fad_type_t<PhysicsType, GlobalStateT, ConfigT>;
        using ElasticStrainT = typename Plato::fad_type_t<PhysicsType, LocalStateT, ConfigT, GlobalStateT>;

        // Functors used to compute residual-related quantities
        Plato::ScalarGrad<mSpaceDim> tComputeScalarGrad;
        Plato::ComputeGradientWorkset<mSpaceDim> tComputeGradient;
        Plato::J2PlasticityUtilities<mSpaceDim>  tJ2PlasticityUtils;
        Plato::StrainDivergence <mSpaceDim> tComputeStrainDivergence ;
        Plato::ThermoPlasticityUtilities<mSpaceDim, PhysicsType> tThermoPlasticityUtils;
        Plato::ComputeStabilization<mSpaceDim> tComputeStabilization(mPressureScaling, mElasticShearModulus);
        Plato::InterpolateFromNodal<mSpaceDim, mNumDofsPerNode, mPressureDofOffset> tInterpolatePressureFromNodal;
        Plato::InterpolateFromNodal<mSpaceDim, mSpaceDim, 0 /*dof offset*/, mSpaceDim> tInterpolatePressGradFromNodal;

        // Residual evaulation functors
        Plato::PressureDivergence<mSpaceDim, mNumDofsPerNode> tPressureDivergence;
        Plato::StressDivergence<mSpaceDim, mNumDofsPerNode, mMechDofOffset> tStressDivergence;
        Plato::ProjectToNode<mSpaceDim, mNumDofsPerNode, mPressureDofOffset> tProjectVolumeStrain;
        Plato::FluxDivergence<mSpaceDim, mNumDofsPerNode, mPressureDofOffset> tStabilizedDivergence;
        Plato::MSIMP tPenaltyFunction(mElasticPropertiesPenaltySIMP, mElasticPropertiesMinErsatzSIMP);

        Plato::ScalarVectorT<ResultT> tPressure("L2 pressure", tNumCells);
        Plato::ScalarVectorT<ConfigT> tCellVolume("cell volume", tNumCells);
        Plato::ScalarVectorT<ResultT> tVolumeStrain("volume strain", tNumCells);
        Plato::ScalarVectorT<ResultT> tStrainDivergence("strain divergence", tNumCells);
        Plato::ScalarMultiVectorT<ResultT> tStabilization("cell stabilization", tNumCells, mSpaceDim);
        Plato::ScalarMultiVectorT<GradScalarT> tPressureGrad("pressure gradient", tNumCells, mSpaceDim);
        Plato::ScalarMultiVectorT<ResultT> tDeviatoricStress("deviatoric stress", tNumCells, mNumVoigtTerms);
        Plato::ScalarMultiVectorT<ElasticStrainT> tElasticStrain("elastic strain", tNumCells, mNumVoigtTerms);
        Plato::ScalarArray3DT<ConfigT> tConfigurationGradient("configuration gradient", tNumCells, mNumNodesPerCell, mSpaceDim);
        Plato::ScalarMultiVectorT<NodeStateT> tProjectedPressureGradGP("projected pressure gradient", tNumCells, mSpaceDim);

        // Transfer elasticity parameters to device
        auto tNumDofsPerNode = mNumDofsPerNode;
        auto tPressureScaling = mPressureScaling;
        auto tPressureDofOffset = mPressureDofOffset;
        auto tElasticBulkModulus = mElasticBulkModulus;
        auto tElasticShearModulus = mElasticShearModulus;

        auto tQuadratureWeight = mCubatureRule->getCubWeight();
        auto tBasisFunctions = mCubatureRule->getBasisFunctions();
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType &aCellOrdinal)
        {
            // compute configuration gradients
            tComputeGradient(aCellOrdinal, tConfigurationGradient, aConfig, tCellVolume);
            tCellVolume(aCellOrdinal) *= tQuadratureWeight;

            // compute elastic strain, i.e. e_elastic = e_total - e_plastic
            tThermoPlasticityUtils.computeElasticStrain(aCellOrdinal, aCurrentGlobalState, aCurrentLocalState,
                                                        tBasisFunctions, tConfigurationGradient, tElasticStrain);

            // compute pressure gradient
            tComputeScalarGrad(aCellOrdinal, tNumDofsPerNode, tPressureDofOffset,
                               aCurrentGlobalState, tConfigurationGradient, tPressureGrad);

            // interpolate projected pressure grad, pressure, and temperature to gauss point
            tInterpolatePressureFromNodal(aCellOrdinal, tBasisFunctions, aCurrentGlobalState, tPressure);
            tInterpolatePressGradFromNodal(aCellOrdinal, tBasisFunctions, aProjectedPressureGrad, tProjectedPressureGradGP);

            // compute cell penalty
            ControlT tDensity = Plato::cell_density<mNumNodesPerCell>(aCellOrdinal, aControl);
            ControlT tElasticPropertiesPenalty = tPenaltyFunction(tDensity);

            // compute deviatoric stress and displacement divergence
            ControlT tPenalizedShearModulus = tElasticPropertiesPenalty * tElasticShearModulus;
            tJ2PlasticityUtils.computeDeviatoricStress(aCellOrdinal, tElasticStrain, tPenalizedShearModulus, tDeviatoricStress);
            //printf("DevStress(%d,1) = %.10f\n", aCellOrdinal, tDeviatoricStress(aCellOrdinal,0));
            //printf("DevStress(%d,2) = %.10f\n", aCellOrdinal, tDeviatoricStress(aCellOrdinal,1));
            //printf("DevStress(%d,3) = %.10f\n", aCellOrdinal, tDeviatoricStress(aCellOrdinal,2));
            tComputeStrainDivergence(aCellOrdinal, tElasticStrain, tStrainDivergence);

            // compute volume difference
            tPressure(aCellOrdinal) *= tPressureScaling * tElasticPropertiesPenalty;
            tVolumeStrain(aCellOrdinal) = tPressureScaling * tElasticPropertiesPenalty
                * (tStrainDivergence(aCellOrdinal) - tPressure(aCellOrdinal) / tElasticBulkModulus);

            // compute cell stabilization term
            tComputeStabilization(aCellOrdinal, tCellVolume, tPressureGrad, tProjectedPressureGradGP, tStabilization);
            Plato::apply_penalty<mSpaceDim>(aCellOrdinal, tElasticPropertiesPenalty, tStabilization);

            // compute residual
            tStressDivergence (aCellOrdinal, aResult, tDeviatoricStress, tConfigurationGradient, tCellVolume);
            tPressureDivergence (aCellOrdinal, aResult, tPressure, tConfigurationGradient, tCellVolume);
            tStabilizedDivergence (aCellOrdinal, aResult, tStabilization, tConfigurationGradient, tCellVolume, -1.0);
            tProjectVolumeStrain (aCellOrdinal, tCellVolume, tBasisFunctions, tVolumeStrain, aResult);
        }, "stabilized elasto-plastic residual");

        this->addExternalForces(aCurrentGlobalState, aControl, aResult);
        this->outputData(tDeviatoricStress, "deviatoric stress");
        this->outputData(tPressure, "pressure");
    }
};
// class StabilizedElastoPlasticResidual







namespace ElastoPlasticityFactory
{

/***************************************************************************//**
 * \brief Factory for elasto-plasticity vector and scalar functions
*******************************************************************************/
struct FunctionFactory
{
    /***************************************************************************//**
     * \brief Create a PLATO local vector function  inc (i.e. local residual equations)
     * \param [in] aMesh mesh database
     * \param [in] aMeshSets side sets database
     * \param [in] aDataMap output data database
     * \param [in] aInputParams input parameters
     * \param [in] aFunctionName vector function name
     * \return shared pointer to a stabilized vector function integrated in time
    *******************************************************************************/
    template<typename EvaluationType>
    std::shared_ptr<Plato::AbstractVectorFunctionVMSInc<EvaluationType>>
    createVectorFunctionVMSInc(Omega_h::Mesh& aMesh, Omega_h::MeshSets& aMeshSets, Plato::DataMap& aDataMap, Teuchos::ParameterList& aInputParams, std::string aFunctionName)
    {
        if(aFunctionName == "ElastoPlasticity")
        {
            constexpr auto tSpaceDim = EvaluationType::SpatialDim;
            return std::make_shared<Plato::ElastoPlasticityResidual<EvaluationType, Plato::SimplexPlasticity<tSpaceDim>> > (aMesh, aMeshSets, aDataMap, aInputParams);
        }
        else
        {
            const std::string tError = std::string("Unknown createVectorFunctionVMSInc '") + aFunctionName + "' specified.";
            THROWERR(tError)
        }
    }
};
// struct FunctionFactory

}
// namespace ElastoPlasticityFactory

/*************************************************************************//**
 * \brief Concrete class defining the Physics Type template argument for a
 * VectorFunctionVMSInc.  A VectorFunctionVMSInc is defined by a stabilized
 * Partial Differential Equation (PDE) integrated implicitly in time.  The
 * stabilization technique is based on the Variational Multiscale (VMS) method.
 * Here, the (Inc) in VectorFunctionVMSInc denotes increment.
*****************************************************************************/
template<Plato::OrdinalType NumSpaceDim>
class ElastoPlasticity: public Plato::SimplexPlasticity<NumSpaceDim>
{
public:
    static constexpr auto mSpaceDim = NumSpaceDim;                           /*!< number of spatial dimensitons */
    typedef Plato::ElastoPlasticityFactory::FunctionFactory FunctionFactory; /*!< define short name for elastoplasticity factory */

    using SimplexT = Plato::SimplexPlasticity<NumSpaceDim>; /*!< define short name for simplex plasticity physics */
    /*!< define short name for projected pressure gradient physics */
    using ProjectorT = typename Plato::Projection<NumSpaceDim, SimplexT::mNumDofsPerNode, SimplexT::mPressureDofOffset>;
};
// class ElastoPlasticity






template<typename PhysicsT>
class VectorFunctionVMSInc
{
// Private access member data
private:
    using Residual        = typename Plato::Evaluation<PhysicsT>::Residual;       /*!< automatic differentiation (AD) type for the residual */
    using GradientX       = typename Plato::Evaluation<PhysicsT>::GradientX;      /*!< AD type for the configuration */
    using GradientZ       = typename Plato::Evaluation<PhysicsT>::GradientZ;      /*!< AD type for the controls */
    using JacobianPgrad   = typename Plato::Evaluation<PhysicsT>::JacobianN;      /*!< AD type for the nodal pressure gradient */
    using LocalJacobian   = typename Plato::Evaluation<PhysicsT>::LocalJacobian;  /*!< AD type for the current local states */
    using LocalJacobianP  = typename Plato::Evaluation<PhysicsT>::LocalJacobianP; /*!< AD type for the previous local states */
    using GlobalJacobian  = typename Plato::Evaluation<PhysicsT>::Jacobian;       /*!< AD type for the current global states */
    using GlobalJacobianP = typename Plato::Evaluation<PhysicsT>::JacobianP;      /*!< AD type for the previous global states */

    static constexpr auto mNumControl = PhysicsT::mNumControl;                        /*!< number of control fields, i.e. vectors, number of materials */
    static constexpr auto mNumSpatialDims = PhysicsT::mNumSpatialDims;                /*!< number of spatial dimensions */
    static constexpr auto mNumNodesPerCell = PhysicsT::mNumNodesPerCell;              /*!< number of nodes per cell (i.e. element) */
    static constexpr auto mNumGlobalDofsPerNode = PhysicsT::mNumDofsPerNode;          /*!< number of global degrees of freedom per node */
    static constexpr auto mNumGlobalDofsPerCell = PhysicsT::mNumDofsPerCell;          /*!< number of global degrees of freedom per cell (i.e. element) */
    static constexpr auto mNumLocalDofsPerCell = PhysicsT::mNumLocalDofsPerCell;      /*!< number of local degrees of freedom per cell (i.e. element) */
    static constexpr auto mNumNodeStatePerNode = PhysicsT::mNumNodeStatePerNode;      /*!< number of pressure gradient degrees of freedom per node */
    static constexpr auto mNumNodeStatePerCell = PhysicsT::mNumNodeStatePerCell;      /*!< number of pressure gradient degrees of freedom per cell (i.e. element) */
    static constexpr auto mNumConfigDofsPerCell = mNumSpatialDims * mNumNodesPerCell; /*!< number of configuration (i.e. coordinates) degrees of freedom per cell (i.e. element) */

    const Plato::OrdinalType mNumNodes; /*!< total number of nodes */
    const Plato::OrdinalType mNumCells; /*!< total number of cells (i.e. elements)*/

    Plato::DataMap& mDataMap;                  /*!< output data map */
    Plato::WorksetBase<PhysicsT> mWorksetBase; /*!< assembly routine interface */

    std::shared_ptr<Plato::AbstractVectorFunctionVMSInc<Residual>>        mGlobalVecFuncResidual;   /*!< global vector function residual */
    std::shared_ptr<Plato::AbstractVectorFunctionVMSInc<GradientX>>       mGlobalVecFuncJacobianX;  /*!< global vector function Jacobian wrt configuration */
    std::shared_ptr<Plato::AbstractVectorFunctionVMSInc<GradientZ>>       mGlobalVecFuncJacobianZ;  /*!< global vector function Jacobian wrt controls */
    std::shared_ptr<Plato::AbstractVectorFunctionVMSInc<JacobianPgrad>>   mGlobalVecFuncJacPgrad;   /*!< global vector function Jacobian wrt projected pressure gradient */
    std::shared_ptr<Plato::AbstractVectorFunctionVMSInc<LocalJacobian>>   mGlobalVecFuncJacobianC;  /*!< global vector function Jacobian wrt current local states */
    std::shared_ptr<Plato::AbstractVectorFunctionVMSInc<LocalJacobianP>>  mGlobalVecFuncJacobianCP; /*!< global vector function Jacobian wrt previous local states */
    std::shared_ptr<Plato::AbstractVectorFunctionVMSInc<GlobalJacobian>>  mGlobalVecFuncJacobianU;  /*!< global vector function Jacobian wrt current global states */
    std::shared_ptr<Plato::AbstractVectorFunctionVMSInc<GlobalJacobianP>> mGlobalVecFuncJacobianUP; /*!< global vector function Jacobian wrt previous global states */

// Public access functions
public:
    /***********************************************************************//**
     * \brief Constructor
     * \param [in] aMesh mesh data base
     * \param [in] aMeshSets mesh sets data base
     * \param [in] aDataMap problem-specific data map
     * \param [in] aParamList Teuchos parameter list with input data
     * \param [in] aVectorFuncType vector function type string
    ***************************************************************************/
    VectorFunctionVMSInc(Omega_h::Mesh& aMesh,
                         Omega_h::MeshSets& aMeshSets,
                         Plato::DataMap& aDataMap,
                         Teuchos::ParameterList& aParamList,
                         std::string& aVectorFuncType) :
            mNumNodes(aMesh.nverts()),
            mNumCells(aMesh.nelems()),
            mDataMap(aDataMap),
            mWorksetBase(aMesh)
    {
        typename PhysicsT::FunctionFactory tFunctionFactory;

        mGlobalVecFuncResidual = tFunctionFactory.template createVectorFunctionVMSInc<Residual>
                                                           (aMesh, aMeshSets, aDataMap, aParamList, aVectorFuncType);

        mGlobalVecFuncJacobianU = tFunctionFactory.template createVectorFunctionVMSInc<GlobalJacobian>
                                                            (aMesh, aMeshSets, aDataMap, aParamList, aVectorFuncType);

        mGlobalVecFuncJacobianUP = tFunctionFactory.template createVectorFunctionVMSInc<GlobalJacobianP>
                                                             (aMesh, aMeshSets, aDataMap, aParamList, aVectorFuncType);

        mGlobalVecFuncJacobianC = tFunctionFactory.template createVectorFunctionVMSInc<LocalJacobian>
                                                            (aMesh, aMeshSets, aDataMap, aParamList, aVectorFuncType);

        mGlobalVecFuncJacobianCP = tFunctionFactory.template createVectorFunctionVMSInc<LocalJacobianP>
                                                             (aMesh, aMeshSets, aDataMap, aParamList, aVectorFuncType);

        mGlobalVecFuncJacobianZ = tFunctionFactory.template createVectorFunctionVMSInc<GradientZ>
                                                            (aMesh, aMeshSets, aDataMap, aParamList, aVectorFuncType);

        mGlobalVecFuncJacobianX = tFunctionFactory.template createVectorFunctionVMSInc<GradientX>
                                                            (aMesh, aMeshSets, aDataMap, aParamList, aVectorFuncType);

        mGlobalVecFuncJacPgrad = tFunctionFactory.template createVectorFunctionVMSInc<JacobianPgrad>
                                                           (aMesh, aMeshSets, aDataMap, aParamList, aVectorFuncType);
    }

    /***********************************************************************//**
     * \brief Destructor
    ***************************************************************************/
    ~VectorFunctionVMSInc(){ return; }

    /***********************************************************************//**
     * \brief Return total number of degrees of freedom
    ***************************************************************************/
    Plato::OrdinalType size() const
    {
        return mNumNodes * mNumGlobalDofsPerNode;
    }

    /***********************************************************************//**
     * \brief Return total number of nodes
     * \return total number of nodes
    ***************************************************************************/
    Plato::OrdinalType numNodes() const
    {
        return mNumNodes;
    }

    /***********************************************************************//**
     * \brief Return total number of cells
     * \return total number of cells
    ***************************************************************************/
    Plato::OrdinalType numCells() const
    {
        return mNumCells;
    }

    /***********************************************************************//**
     * \brief Compute the global residual vector
     * \param [in] aGlobalState global state at current time step
     * \param [in] aPrevGlobalState global state at previous time step
     * \param [in] aLocalState local state at current time step
     * \param [in] aPrevLocalState local state at previous time step
     * \param [in] aControl control parameters
     * \param [in] aNodeState pressure gradient
     * \param [in] aTimeStep time step
     * \return Assembled global residual vector
    ***************************************************************************/
    Plato::ScalarVector
    value(const Plato::ScalarVector & aGlobalState,
          const Plato::ScalarVector & aPrevGlobalState,
          const Plato::ScalarVector & aLocalState,
          const Plato::ScalarVector & aPrevLocalState,
          const Plato::ScalarVector & aNodeState,
          const Plato::ScalarVector & aControl,
          Plato::Scalar aTimeStep = 0.0) const
    {
        // Workset config
        using ConfigScalar = typename Residual::ConfigScalarType;
        Plato::ScalarArray3DT<ConfigScalar>
            tConfigWS("Config Workset", mNumCells, mNumNodesPerCell, mNumSpatialDims);
        mWorksetBase.worksetConfig(tConfigWS);

        // Workset current global state
        using GlobalStateScalar = typename Residual::StateScalarType;
        Plato::ScalarMultiVectorT<GlobalStateScalar>
            tGlobalStateWS("Current Global State Workset", mNumCells, mNumGlobalDofsPerCell);
        mWorksetBase.worksetState(aGlobalState, tGlobalStateWS);

        // Workset previous global state
        using PrevGlobalStateScalar = typename Residual::PrevStateScalarType;
        Plato::ScalarMultiVectorT<PrevGlobalStateScalar>
            tPrevGlobalStateWS("Previous Global State Workset", mNumCells, mNumGlobalDofsPerCell);
        mWorksetBase.worksetState(aPrevGlobalState, tPrevGlobalStateWS);

        // Workset local state
        using LocalStateScalar = typename Residual::LocalStateScalarType;
        Plato::ScalarMultiVectorT<LocalStateScalar>
            tLocalStateWS("Local Current State Workset", mNumCells, mNumLocalDofsPerCell);
        mWorksetBase.worksetLocalState(aLocalState, tLocalStateWS);

        // Workset previous local state
        using PrevLocalStateScalar = typename Residual::PrevLocalStateScalarType;
        Plato::ScalarMultiVectorT<PrevLocalStateScalar>
            tPrevLocalStateWS("Previous Local State Workset", mNumCells, mNumLocalDofsPerCell);
        mWorksetBase.worksetLocalState(aPrevLocalState, tPrevLocalStateWS);

        // Workset node state, i.e. projected pressure gradient
        using NodeStateScalar = typename Residual::NodeStateScalarType;
        Plato::ScalarMultiVectorT<NodeStateScalar> tNodeStateWS("Node State Workset", mNumCells, mNumNodeStatePerCell);
        mWorksetBase.worksetNodeState(aNodeState, tNodeStateWS);

        // Workset control
        using ControlScalar = typename Residual::ControlScalarType;
        Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", mNumCells, mNumNodesPerCell);
        mWorksetBase.worksetControl(aControl, tControlWS);

        // Workset residual
        using ResultScalar = typename Residual::ResultScalarType;
        Plato::ScalarMultiVectorT<ResultScalar> tResidualWS("Residual Workset", mNumCells, mNumGlobalDofsPerCell);

        // Evaluate global residual
        mGlobalVecFuncResidual->evaluate(tGlobalStateWS, tPrevGlobalStateWS, tLocalStateWS, tPrevLocalStateWS,
                                         tNodeStateWS, tControlWS, tConfigWS, tResidualWS, aTimeStep);

        // create and assemble to return view
        const auto tTotalNumDofs = mNumGlobalDofsPerNode * mNumNodes;
        Kokkos::View<Plato::Scalar*, Kokkos::LayoutRight, Plato::MemSpace>  tAssembledResidual("Assembled Residual", tTotalNumDofs);
        mWorksetBase.assembleResidual( tResidualWS, tAssembledResidual );

        return tAssembledResidual;
    }

    /***********************************************************************//**
     * \brief Compute Jacobian with respect to (wrt) control of the global residual
     * \param [in] aGlobalState global state at current time step
     * \param [in] aPrevGlobalState global state at previous time step
     * \param [in] aLocalState local state at current time step
     * \param [in] aPrevLocalState local state at previous time step
     * \param [in] aControl control parameters
     * \param [in] aNodeState pressure gradient
     * \param [in] aTimeStep time step
     * \return Jacobian wrt control of the global residual
    ***************************************************************************/
    Teuchos::RCP<Plato::CrsMatrixType>
    gradient_z(const Plato::ScalarVector & aGlobalState,
               const Plato::ScalarVector & aPrevGlobalState,
               const Plato::ScalarVector & aLocalState,
               const Plato::ScalarVector & aPrevLocalState,
               const Plato::ScalarVector & aNodeState,
               const Plato::ScalarVector & aControl,
               Plato::Scalar aTimeStep = 0.0) const
    {
        // Workset config
        using ConfigScalar = typename GradientZ::ConfigScalarType;
        Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset", mNumCells, mNumNodesPerCell, mNumSpatialDims);
        mWorksetBase.worksetConfig(tConfigWS);

        // Workset current global state
        using GlobalStateScalar = typename GradientZ::StateScalarType;
        Plato::ScalarMultiVectorT<GlobalStateScalar> tGlobalStateWS("Current Global State Workset", mNumCells, mNumGlobalDofsPerCell);
        mWorksetBase.worksetState(aGlobalState, tGlobalStateWS);

        // Workset previous global state
        using PrevGlobalStateScalar = typename GradientZ::PrevStateScalarType;
        Plato::ScalarMultiVectorT<PrevGlobalStateScalar> tPrevGlobalStateWS("Previous Global State Workset", mNumCells, mNumGlobalDofsPerCell);
        mWorksetBase.worksetState(aPrevGlobalState, tPrevGlobalStateWS);

        // Workset local state
        using LocalStateScalar = typename GradientZ::LocalStateScalarType;
        Plato::ScalarMultiVectorT<LocalStateScalar> tLocalStateWS("Local Current State Workset", mNumCells, mNumLocalDofsPerCell);
        mWorksetBase.worksetLocalState(aLocalState, tLocalStateWS);

        // Workset previous local state
        using PrevLocalStateScalar = typename GradientZ::PrevLocalStateScalarType;
        Plato::ScalarMultiVectorT<PrevLocalStateScalar> tPrevLocalStateWS("Previous Local State Workset", mNumCells, mNumLocalDofsPerCell);
        mWorksetBase.worksetLocalState(aPrevLocalState, tPrevLocalStateWS);

        // Workset node state, i.e. projected pressure gradient
        using NodeStateScalar = typename GradientZ::NodeStateScalarType;
        Plato::ScalarMultiVectorT<NodeStateScalar> tNodeStateWS("Node State Workset", mNumCells, mNumNodeStatePerCell);
        mWorksetBase.worksetNodeState(aNodeState, tNodeStateWS);

        // Workset control
        using ControlScalar = typename GradientZ::ControlScalarType;
        Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", mNumCells, mNumNodesPerCell);
        mWorksetBase.worksetControl(aControl, tControlWS);

        // Create Jacobian workset
        using JacobianScalar = typename GradientZ::ResultScalarType;
        Plato::ScalarMultiVectorT<JacobianScalar> tJacobianWS("Jacobian Control Workset", mNumCells, mNumGlobalDofsPerCell);

        // Call evaluate function - compute Jacobian wrt controls
        mGlobalVecFuncJacobianZ->evaluate(tGlobalStateWS, tPrevGlobalStateWS, tLocalStateWS, tPrevLocalStateWS,
                                          tNodeStateWS, tControlWS, tConfigWS, tJacobianWS, aTimeStep);

        // Create return Jacobain
        auto tMesh = mGlobalVecFuncJacobianZ->getMesh();
        Teuchos::RCP<Plato::CrsMatrixType> tAssembledJacobian =
                Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumControl, mNumGlobalDofsPerNode>(&tMesh);

        // Assemble Jacobian
        Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumControl, mNumGlobalDofsPerNode>
                tJacobianMatEntryOrdinal(tAssembledJacobian, &tMesh);
        auto tJacobianMatEntries = tAssembledJacobian->entries();
        mWorksetBase.assembleTransposeJacobian(mNumGlobalDofsPerCell, mNumNodesPerCell,
                                               tJacobianMatEntryOrdinal, tJacobianWS, tJacobianMatEntries);

        return tAssembledJacobian;
    }

    /***********************************************************************//**
     * \brief Compute Jacobian wrt configuration of the global residual
     * \param [in] aGlobalState global state at current time step
     * \param [in] aPrevGlobalState global state at previous time step
     * \param [in] aLocalState local state at current time step
     * \param [in] aPrevLocalState local state at previous time step
     * \param [in] aControl control parameters
     * \param [in] aNodeState pressure gradient
     * \param [in] aTimeStep time step
     * \return Jacobian wrt configuration of the global residual
    ***************************************************************************/
    Teuchos::RCP<Plato::CrsMatrixType>
    gradient_x(const Plato::ScalarVector & aGlobalState,
               const Plato::ScalarVector & aPrevGlobalState,
               const Plato::ScalarVector & aLocalState,
               const Plato::ScalarVector & aPrevLocalState,
               const Plato::ScalarVector & aNodeState,
               const Plato::ScalarVector & aControl,
               Plato::Scalar aTimeStep = 0.0) const
    {
        // Workset config
        using ConfigScalar = typename GradientX::ConfigScalarType;
        Plato::ScalarArray3DT<ConfigScalar>
            tConfigWS("Config Workset", mNumCells, mNumNodesPerCell, mNumSpatialDims);
        mWorksetBase.worksetConfig(tConfigWS);

        // Workset current global state
        using GlobalStateScalar = typename GradientX::StateScalarType;
        Plato::ScalarMultiVectorT<GlobalStateScalar>
            tGlobalStateWS("Current Global State Workset", mNumCells, mNumGlobalDofsPerCell);
        mWorksetBase.worksetState(aGlobalState, tGlobalStateWS);

        // Workset previous global state
        using PrevGlobalStateScalar = typename GradientX::PrevStateScalarType;
        Plato::ScalarMultiVectorT<PrevGlobalStateScalar>
            tPrevGlobalStateWS("Previous Global State Workset", mNumCells, mNumGlobalDofsPerCell);
        mWorksetBase.worksetState(aPrevGlobalState, tPrevGlobalStateWS);

        // Workset local state
        using LocalStateScalar = typename GradientX::LocalStateScalarType;
        Plato::ScalarMultiVectorT<LocalStateScalar>
            tLocalStateWS("Local Current State Workset", mNumCells, mNumLocalDofsPerCell);
        mWorksetBase.worksetLocalState(aLocalState, tLocalStateWS);

        // Workset previous local state
        using PrevLocalStateScalar = typename GradientX::PrevLocalStateScalarType;
        Plato::ScalarMultiVectorT<PrevLocalStateScalar>
            tPrevLocalStateWS("Previous Local State Workset", mNumCells, mNumLocalDofsPerCell);
        mWorksetBase.worksetLocalState(aPrevLocalState, tPrevLocalStateWS);

        // Workset node state, i.e. projected pressure gradient
        using NodeStateScalar = typename GradientX::NodeStateScalarType;
        Plato::ScalarMultiVectorT<NodeStateScalar> tNodeStateWS("Node State Workset", mNumCells, mNumNodeStatePerCell);
        mWorksetBase.worksetNodeState(aNodeState, tNodeStateWS);

        // Workset control
        using ControlScalar = typename GradientX::ControlScalarType;
        Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", mNumCells, mNumNodesPerCell);
        mWorksetBase.worksetControl(aControl, tControlWS);

        // create return view
        using JacobianScalar = typename GradientX::ResultScalarType;
        Plato::ScalarMultiVectorT<JacobianScalar> tJacobianWS("Jacobian Configuration", mNumCells, mNumGlobalDofsPerCell);

        // Call evaluate function - compute Jacobian wrt configuration
        mGlobalVecFuncJacobianX->evaluate(tGlobalStateWS, tPrevGlobalStateWS, tLocalStateWS, tPrevLocalStateWS,
                                          tNodeStateWS, tControlWS, tConfigWS, tJacobianWS, aTimeStep);

        // create return matrix
        auto tMesh = mGlobalVecFuncJacobianX->getMesh();
        Teuchos::RCP<Plato::CrsMatrixType> tAssembledJacobian =
                Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumSpatialDims, mNumGlobalDofsPerNode>(&tMesh);

        // Assemble Jacobian
        Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumSpatialDims, mNumGlobalDofsPerNode>
                tJacobianEntryOrdinal(tAssembledJacobian, &tMesh);

        auto tJacobianMatEntries = tAssembledJacobian->entries();
        mWorksetBase.assembleTransposeJacobian(mNumGlobalDofsPerCell, mNumConfigDofsPerCell,
                                               tJacobianEntryOrdinal, tJacobianWS, tJacobianMatEntries);

        return tAssembledJacobian;
    }

    /***********************************************************************//**
     * \brief Compute Jacobian wrt current global states of the global residual
     * \param [in] aGlobalState global state at current time step
     * \param [in] aPrevGlobalState global state at previous time step
     * \param [in] aLocalState local state at current time step
     * \param [in] aPrevLocalState local state at previous time step
     * \param [in] aControl control parameters
     * \param [in] aNodeState pressure gradient
     * \param [in] aTimeStep time step
     * \return Jacobian wrt current global states of the global residual
    ***************************************************************************/
    Plato::ScalarArray3D
    gradient_u(const Plato::ScalarVector & aGlobalState,
               const Plato::ScalarVector & aPrevGlobalState,
               const Plato::ScalarVector & aLocalState,
               const Plato::ScalarVector & aPrevLocalState,
               const Plato::ScalarVector & aNodeState,
               const Plato::ScalarVector & aControl,
               Plato::Scalar aTimeStep = 0.0) const
    {
        // Workset config
        using ConfigScalar = typename GlobalJacobian::ConfigScalarType;
        Plato::ScalarArray3DT<ConfigScalar>
            tConfigWS("Config Workset", mNumCells, mNumNodesPerCell, mNumSpatialDims);
        mWorksetBase.worksetConfig(tConfigWS);

        // Workset current global state
        using GlobalStateScalar = typename GlobalJacobian::StateScalarType;
        Plato::ScalarMultiVectorT<GlobalStateScalar>
            tGlobalStateWS("Current Global State Workset", mNumCells, mNumGlobalDofsPerCell);
        mWorksetBase.worksetState(aGlobalState, tGlobalStateWS);

        // Workset previous global state
        using PrevGlobalStateScalar = typename GlobalJacobian::PrevStateScalarType;
        Plato::ScalarMultiVectorT<PrevGlobalStateScalar>
            tPrevGlobalStateWS("Previous Global State Workset", mNumCells, mNumGlobalDofsPerCell);
        mWorksetBase.worksetState(aPrevGlobalState, tPrevGlobalStateWS);

        // Workset local state
        using LocalStateScalar = typename GlobalJacobian::LocalStateScalarType;
        Plato::ScalarMultiVectorT<LocalStateScalar>
            tLocalStateWS("Local Current State Workset", mNumCells, mNumLocalDofsPerCell);
        mWorksetBase.worksetLocalState(aLocalState, tLocalStateWS);

        // Workset previous local state
        using PrevLocalStateScalar = typename GlobalJacobian::PrevLocalStateScalarType;
        Plato::ScalarMultiVectorT<PrevLocalStateScalar>
            tPrevLocalStateWS("Previous Local State Workset", mNumCells, mNumLocalDofsPerCell);
        mWorksetBase.worksetLocalState(aPrevLocalState, tPrevLocalStateWS);

        // Workset node state, i.e. projected pressure gradient
        using NodeStateScalar = typename GlobalJacobian::NodeStateScalarType;
        Plato::ScalarMultiVectorT<NodeStateScalar> tNodeStateWS("Node State Workset", mNumCells, mNumNodeStatePerCell);
        mWorksetBase.worksetNodeState(aNodeState, tNodeStateWS);

        // Workset control
        using ControlScalar = typename GlobalJacobian::ControlScalarType;
        Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", mNumCells, mNumNodesPerCell);
        mWorksetBase.worksetControl(aControl, tControlWS);

        // Workset Jacobian wrt current global states
        using JacobianScalar = typename GlobalJacobian::ResultScalarType;
        Plato::ScalarMultiVectorT<JacobianScalar> tJacobianWS("Jacobian Current Global State", mNumCells, mNumGlobalDofsPerCell);

        // Call evaluate function - compute Jacobian wrt the current global states
        mGlobalVecFuncJacobianU->evaluate(tGlobalStateWS, tPrevGlobalStateWS, tLocalStateWS, tPrevLocalStateWS,
                                          tNodeStateWS, tControlWS, tConfigWS, tJacobianWS, aTimeStep);

        Plato::ScalarArray3D tOutputJacobian("Output Jacobian Current State", mNumCells, mNumGlobalDofsPerCell, mNumGlobalDofsPerCell);
        Plato::convert_ad_type_to_scalar_type<mNumGlobalDofsPerCell, mNumGlobalDofsPerCell>(mNumCells, tJacobianWS, tOutputJacobian);
        return tOutputJacobian;
    }

    /***********************************************************************//**
     * \brief Compute transpose Jacobian wrt current global states of the global residual
     * \param [in] aGlobalState global state at current time step
     * \param [in] aPrevGlobalState global state at previous time step
     * \param [in] aLocalState local state at current time step
     * \param [in] aPrevLocalState local state at previous time step
     * \param [in] aControl control parameters
     * \param [in] aNodeState pressure gradient
     * \param [in] aTimeStep time step
     * \return transpose Jacobian wrt current global states of the global residual
    ***************************************************************************/
    Plato::ScalarArray3D
    gradient_u_T(const Plato::ScalarVector & aGlobalState,
                 const Plato::ScalarVector & aPrevGlobalState,
                 const Plato::ScalarVector & aLocalState,
                 const Plato::ScalarVector & aPrevLocalState,
                 const Plato::ScalarVector & aNodeState,
                 const Plato::ScalarVector & aControl,
                 Plato::Scalar aTimeStep = 0.0) const
    {
        // Global residual is symmetric
        return (this->gradient_u(aGlobalState, aPrevGlobalState, aLocalState, aPrevLocalState, aNodeState, aControl, aTimeStep));
    }

    /***********************************************************************//**
     * \brief Compute Jacobian wrt previous global states of the global residual
     * \param [in] aGlobalState global state at current time step
     * \param [in] aPrevGlobalState global state at previous time step
     * \param [in] aLocalState local state at current time step
     * \param [in] aPrevLocalState local state at previous time step
     * \param [in] aControl control parameters
     * \param [in] aNodeState pressure gradient
     * \param [in] aTimeStep time step
     * \return Jacobian wrt previous global states of the global residual
    ***************************************************************************/
    Plato::ScalarArray3D
    gradient_up(const Plato::ScalarVector & aGlobalState,
                const Plato::ScalarVector & aPrevGlobalState,
                const Plato::ScalarVector & aLocalState,
                const Plato::ScalarVector & aPrevLocalState,
                const Plato::ScalarVector & aNodeState,
                const Plato::ScalarVector & aControl,
                Plato::Scalar aTimeStep = 0.0) const
    {
        // Workset config
        using ConfigScalar = typename GlobalJacobianP::ConfigScalarType;
        Plato::ScalarArray3DT<ConfigScalar>
            tConfigWS("Config Workset", mNumCells, mNumNodesPerCell, mNumSpatialDims);
        mWorksetBase.worksetConfig(tConfigWS);

        // Workset current global state
        using GlobalStateScalar = typename GlobalJacobianP::StateScalarType;
        Plato::ScalarMultiVectorT<GlobalStateScalar>
            tGlobalStateWS("Current Global State Workset", mNumCells, mNumGlobalDofsPerCell);
        mWorksetBase.worksetState(aGlobalState, tGlobalStateWS);

        // Workset previous global state
        using PrevGlobalStateScalar = typename GlobalJacobianP::PrevStateScalarType;
        Plato::ScalarMultiVectorT<PrevGlobalStateScalar>
            tPrevGlobalStateWS("Previous Global State Workset", mNumCells, mNumGlobalDofsPerCell);
        mWorksetBase.worksetState(aPrevGlobalState, tPrevGlobalStateWS);

        // Workset local state
        using LocalStateScalar = typename GlobalJacobianP::LocalStateScalarType;
        Plato::ScalarMultiVectorT<LocalStateScalar>
            tLocalStateWS("Local Current State Workset", mNumCells, mNumLocalDofsPerCell);
        mWorksetBase.worksetLocalState(aLocalState, tLocalStateWS);

        // Workset previous local state
        using PrevLocalStateScalar = typename GlobalJacobianP::PrevLocalStateScalarType;
        Plato::ScalarMultiVectorT<PrevLocalStateScalar>
            tPrevLocalStateWS("Previous Local State Workset", mNumCells, mNumLocalDofsPerCell);
        mWorksetBase.worksetLocalState(aPrevLocalState, tPrevLocalStateWS);

        // Workset node state, i.e. projected pressure gradient
        using NodeStateScalar = typename GlobalJacobianP::NodeStateScalarType;
        Plato::ScalarMultiVectorT<NodeStateScalar> tNodeStateWS("Node State Workset", mNumCells, mNumNodeStatePerCell);
        mWorksetBase.worksetNodeState(aNodeState, tNodeStateWS);

        // Workset control
        using ControlScalar = typename GlobalJacobianP::ControlScalarType;
        Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", mNumCells, mNumNodesPerCell);
        mWorksetBase.worksetControl(aControl, tControlWS);

        // Workset Jacobian wrt current global states
        using JacobianScalar = typename GlobalJacobianP::ResultScalarType;
        Plato::ScalarMultiVectorT<JacobianScalar> tJacobianWS("Jacobian Previous Global State", mNumCells, mNumGlobalDofsPerCell);

        // Call evaluate function - compute Jacobian wrt the previous global states
        mGlobalVecFuncJacobianU->evaluate(tGlobalStateWS, tPrevGlobalStateWS, tLocalStateWS, tPrevLocalStateWS,
                                          tNodeStateWS, tControlWS, tConfigWS, tJacobianWS, aTimeStep);

        Plato::ScalarArray3D tOutputJacobian("Output Jacobian Previous Global State", mNumCells, mNumGlobalDofsPerCell, mNumGlobalDofsPerCell);
        Plato::convert_ad_type_to_scalar_type<mNumGlobalDofsPerCell, mNumGlobalDofsPerCell>(mNumCells, tJacobianWS, tOutputJacobian);
        return tOutputJacobian;
    }

    /***********************************************************************//**
     * \brief Compute transpose Jacobian wrt previous global states of the global residual
     * \param [in] aGlobalState global state at current time step
     * \param [in] aPrevGlobalState global state at previous time step
     * \param [in] aLocalState local state at current time step
     * \param [in] aPrevLocalState local state at previous time step
     * \param [in] aControl control parameters
     * \param [in] aNodeState pressure gradient
     * \param [in] aTimeStep time step
     * \return transpose Jacobian wrt previous global states of the global residual
    ***************************************************************************/
    Plato::ScalarArray3D
    gradient_up_T(const Plato::ScalarVector & aGlobalState,
                  const Plato::ScalarVector & aPrevGlobalState,
                  const Plato::ScalarVector & aLocalState,
                  const Plato::ScalarVector & aPrevLocalState,
                  const Plato::ScalarVector & aNodeState,
                  const Plato::ScalarVector & aControl,
                  Plato::Scalar aTimeStep = 0.0) const
    {
        // Global residual is symmetric
        return (this->gradient_up(aGlobalState, aPrevGlobalState, aLocalState, aPrevLocalState, aNodeState, aControl, aTimeStep));
    }

    /***********************************************************************//**
     * \brief Compute Jacobian wrt current local state of the global residual
     * \param [in] aGlobalState global state at current time step
     * \param [in] aPrevGlobalState global state at previous time step
     * \param [in] aLocalState local state at current time step
     * \param [in] aPrevLocalState local state at previous time step
     * \param [in] aControl control parameters
     * \param [in] aNodeState pressure gradient
     * \param [in] aTimeStep time step
     * \return Jacobian wrt current local state of the global residual
    ***************************************************************************/
    Plato::ScalarArray3D
    gradient_c(const Plato::ScalarVector & aGlobalState,
               const Plato::ScalarVector & aPrevGlobalState,
               const Plato::ScalarVector & aLocalState,
               const Plato::ScalarVector & aPrevLocalState,
               const Plato::ScalarVector & aNodeState,
               const Plato::ScalarVector & aControl,
               Plato::Scalar aTimeStep = 0.0) const
    {
        // Workset config
        using ConfigScalar = typename LocalJacobian::ConfigScalarType;
        Plato::ScalarArray3DT<ConfigScalar>
            tConfigWS("Config Workset", mNumCells, mNumNodesPerCell, mNumSpatialDims);
        mWorksetBase.worksetConfig(tConfigWS);

        // Workset current global state
        using GlobalStateScalar = typename LocalJacobian::StateScalarType;
        Plato::ScalarMultiVectorT<GlobalStateScalar>
            tGlobalStateWS("Current Global State Workset", mNumCells, mNumGlobalDofsPerCell);
        mWorksetBase.worksetState(aGlobalState, tGlobalStateWS);

        // Workset previous global state
        using PrevGlobalStateScalar = typename LocalJacobian::PrevStateScalarType;
        Plato::ScalarMultiVectorT<PrevGlobalStateScalar>
            tPrevGlobalStateWS("Previous Global State Workset", mNumCells, mNumGlobalDofsPerCell);
        mWorksetBase.worksetState(aPrevGlobalState, tPrevGlobalStateWS);

        // Workset local state
        using LocalStateScalar = typename LocalJacobian::LocalStateScalarType;
        Plato::ScalarMultiVectorT<LocalStateScalar>
            tLocalStateWS("Local Current State Workset", mNumCells, mNumLocalDofsPerCell);
        mWorksetBase.worksetLocalState(aLocalState, tLocalStateWS);

        // Workset previous local state
        using PrevLocalStateScalar = typename LocalJacobian::PrevLocalStateScalarType;
        Plato::ScalarMultiVectorT<PrevLocalStateScalar>
            tPrevLocalStateWS("Previous Local State Workset", mNumCells, mNumLocalDofsPerCell);
        mWorksetBase.worksetLocalState(aPrevLocalState, tPrevLocalStateWS);

        // Workset node state, i.e. projected pressure gradient
        using NodeStateScalar = typename LocalJacobian::NodeStateScalarType;
        Plato::ScalarMultiVectorT<NodeStateScalar> tNodeStateWS("Node State Workset", mNumCells, mNumNodeStatePerCell);
        mWorksetBase.worksetNodeState(aNodeState, tNodeStateWS);

        // Workset control
        using ControlScalar = typename LocalJacobian::ControlScalarType;
        Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", mNumCells, mNumNodesPerCell);
        mWorksetBase.worksetControl(aControl, tControlWS);

        // Workset Jacobian wrt current local states
        using JacobianScalar = typename LocalJacobian::ResultScalarType;
        Plato::ScalarMultiVectorT<JacobianScalar> tJacobianWS("Jacobian Local State Workset", mNumCells, mNumGlobalDofsPerCell);

        // Call evaluate function - compute Jacobian wrt the current local states
        mGlobalVecFuncJacobianC->evaluate(tGlobalStateWS, tPrevGlobalStateWS, tLocalStateWS, tPrevLocalStateWS,
                                          tNodeStateWS, tControlWS, tConfigWS, tJacobianWS, aTimeStep);

        Plato::ScalarArray3D tOutputJacobian("Output Jacobian Current Local State", mNumCells, mNumGlobalDofsPerCell, mNumLocalDofsPerCell);
        Plato::convert_ad_type_to_scalar_type<mNumGlobalDofsPerCell, mNumLocalDofsPerCell>(mNumCells, tJacobianWS, tOutputJacobian);
        return tOutputJacobian;
    }

    /***********************************************************************//**
     * \brief Compute transpose Jacobian wrt current local states of the global residual
     * \param [in] aGlobalState global state at current time step
     * \param [in] aPrevGlobalState global state at previous time step
     * \param [in] aLocalState local state at current time step
     * \param [in] aPrevLocalState local state at previous time step
     * \param [in] aControl control parameters
     * \param [in] aNodeState pressure gradient
     * \param [in] aTimeStep time step
     * \return transpose Jacobian wrt current local states of the global residual
    ***************************************************************************/
    Plato::ScalarArray3D
    gradient_c_T(const Plato::ScalarVector & aGlobalState,
                 const Plato::ScalarVector & aPrevGlobalState,
                 const Plato::ScalarVector & aLocalState,
                 const Plato::ScalarVector & aPrevLocalState,
                 const Plato::ScalarVector & aNodeState,
                 const Plato::ScalarVector & aControl,
                 Plato::Scalar aTimeStep = 0.0) const
    {
        // Local residual is symmetric
        return (this->gradient_c(aGlobalState, aPrevGlobalState, aLocalState, aPrevLocalState, aNodeState, aControl, aTimeStep));
    }

    /***********************************************************************//**
     * \brief Compute Jacobian wrt previous local state of the global residual
     * \param [in] aGlobalState global state at current time step
     * \param [in] aPrevGlobalState global state at previous time step
     * \param [in] aLocalState local state at current time step
     * \param [in] aPrevLocalState local state at previous time step
     * \param [in] aControl control parameters
     * \param [in] aNodeState pressure gradient
     * \param [in] aTimeStep time step
     * \return Jacobian wrt previous local state of the global residual
    ***************************************************************************/
    Plato::ScalarArray3D
    gradient_cp(const Plato::ScalarVector & aGlobalState,
                const Plato::ScalarVector & aPrevGlobalState,
                const Plato::ScalarVector & aLocalState,
                const Plato::ScalarVector & aPrevLocalState,
                const Plato::ScalarVector & aNodeState,
                const Plato::ScalarVector & aControl,
                Plato::Scalar aTimeStep = 0.0) const
    {
        // Workset config
        using ConfigScalar = typename LocalJacobianP::ConfigScalarType;
        Plato::ScalarArray3DT<ConfigScalar>
            tConfigWS("Config Workset", mNumCells, mNumNodesPerCell, mNumSpatialDims);
        mWorksetBase.worksetConfig(tConfigWS);

        // Workset current global state
        using GlobalStateScalar = typename LocalJacobianP::StateScalarType;
        Plato::ScalarMultiVectorT<GlobalStateScalar>
            tGlobalStateWS("Current Global State Workset", mNumCells, mNumGlobalDofsPerCell);
        mWorksetBase.worksetState(aGlobalState, tGlobalStateWS);

        // Workset previous global state
        using PrevGlobalStateScalar = typename LocalJacobianP::PrevStateScalarType;
        Plato::ScalarMultiVectorT<PrevGlobalStateScalar>
            tPrevGlobalStateWS("Previous Global State Workset", mNumCells, mNumGlobalDofsPerCell);
        mWorksetBase.worksetState(aPrevGlobalState, tPrevGlobalStateWS);

        // Workset local state
        using LocalStateScalar = typename LocalJacobianP::LocalStateScalarType;
        Plato::ScalarMultiVectorT<LocalStateScalar>
            tLocalStateWS("Local Current State Workset", mNumCells, mNumLocalDofsPerCell);
        mWorksetBase.worksetLocalState(aLocalState, tLocalStateWS);

        // Workset previous local state
        using PrevLocalStateScalar = typename LocalJacobianP::PrevLocalStateScalarType;
        Plato::ScalarMultiVectorT<PrevLocalStateScalar>
            tPrevLocalStateWS("Previous Local State Workset", mNumCells, mNumLocalDofsPerCell);
        mWorksetBase.worksetLocalState(aPrevLocalState, tPrevLocalStateWS);

        // Workset node state, i.e. projected pressure gradient
        using NodeStateScalar = typename LocalJacobianP::NodeStateScalarType;
        Plato::ScalarMultiVectorT<NodeStateScalar> tNodeStateWS("Node State Workset", mNumCells, mNumNodeStatePerCell);
        mWorksetBase.worksetNodeState(aNodeState, tNodeStateWS);

        // Workset control
        using ControlScalar = typename LocalJacobianP::ControlScalarType;
        Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", mNumCells, mNumNodesPerCell);
        mWorksetBase.worksetControl(aControl, tControlWS);

        // Workset Jacobian wrt previous local states
        using JacobianScalar = typename LocalJacobianP::ResultScalarType;
        Plato::ScalarMultiVectorT<JacobianScalar> tJacobianWS("Jacobian Previous Local State Workset", mNumCells, mNumGlobalDofsPerCell);

        // Call evaluate function - compute Jacobian wrt the previous local states
        mGlobalVecFuncJacobianCP->evaluate(tGlobalStateWS, tPrevGlobalStateWS, tLocalStateWS, tPrevLocalStateWS,
                                           tNodeStateWS, tControlWS, tConfigWS, tJacobianWS, aTimeStep);

        Plato::ScalarArray3D tOutputJacobian("Output Jacobian Previous Local State", mNumCells, mNumGlobalDofsPerCell, mNumLocalDofsPerCell);
        Plato::convert_ad_type_to_scalar_type<mNumGlobalDofsPerCell, mNumLocalDofsPerCell>(mNumCells, tJacobianWS, tOutputJacobian);
        return tOutputJacobian;
    }

    /***********************************************************************//**
     * \brief Compute transpose Jacobian wrt previous local states of the global residual
     * \param [in] aGlobalState global state at current time step
     * \param [in] aPrevGlobalState global state at previous time step
     * \param [in] aLocalState local state at current time step
     * \param [in] aPrevLocalState local state at previous time step
     * \param [in] aControl control parameters
     * \param [in] aNodeState pressure gradient
     * \param [in] aTimeStep time step
     * \return transpose Jacobian wrt previous local states of the global residual
    ***************************************************************************/
    Plato::ScalarArray3D
    gradient_cp_T(const Plato::ScalarVector & aGlobalState,
                  const Plato::ScalarVector & aPrevGlobalState,
                  const Plato::ScalarVector & aLocalState,
                  const Plato::ScalarVector & aPrevLocalState,
                  const Plato::ScalarVector & aNodeState,
                  const Plato::ScalarVector & aControl,
                  Plato::Scalar aTimeStep = 0.0) const
    {
        // Local residual is symmetric
        return (this->gradient_cp(aGlobalState, aPrevGlobalState, aLocalState, aPrevLocalState, aNodeState, aControl, aTimeStep));
    }

    /***********************************************************************//**
     * \brief Compute assembled Jacobian wrt pressure gradient of the global residual
     * \param [in] aGlobalState global state at current time step
     * \param [in] aPrevGlobalState global state at previous time step
     * \param [in] aLocalState local state at current time step
     * \param [in] aPrevLocalState local state at previous time step
     * \param [in] aControl control parameters
     * \param [in] aNodeState pressure gradient
     * \param [in] aTimeStep time step
     * \return Assembled Jacobian wrt pressure gradient of the global residual
    ***************************************************************************/
    Teuchos::RCP<Plato::CrsMatrixType>
    gradient_n(const Plato::ScalarVector & aGlobalState,
               const Plato::ScalarVector & aPrevGlobalState,
               const Plato::ScalarVector & aLocalState,
               const Plato::ScalarVector & aPrevLocalState,
               const Plato::ScalarVector & aNodeState,
               const Plato::ScalarVector & aControl,
               Plato::Scalar aTimeStep = 0.0) const
    {
        // Workset config
        using ConfigScalar = typename JacobianPgrad::ConfigScalarType;
        Plato::ScalarArray3DT<ConfigScalar>
            tConfigWS("Config Workset", mNumCells, mNumNodesPerCell, mNumSpatialDims);
        mWorksetBase.worksetConfig(tConfigWS);

        // Workset current global state
        using GlobalStateScalar = typename JacobianPgrad::StateScalarType;
        Plato::ScalarMultiVectorT<GlobalStateScalar>
            tGlobalStateWS("Current Global State Workset", mNumCells, mNumGlobalDofsPerCell);
        mWorksetBase.worksetState(aGlobalState, tGlobalStateWS);

        // Workset previous global state
        using PrevGlobalStateScalar = typename JacobianPgrad::PrevStateScalarType;
        Plato::ScalarMultiVectorT<PrevGlobalStateScalar>
            tPrevGlobalStateWS("Previous Global State Workset", mNumCells, mNumGlobalDofsPerCell);
        mWorksetBase.worksetState(aPrevGlobalState, tPrevGlobalStateWS);

        // Workset local state
        using LocalStateScalar = typename JacobianPgrad::LocalStateScalarType;
        Plato::ScalarMultiVectorT<LocalStateScalar>
            tLocalStateWS("Local Current State Workset", mNumCells, mNumLocalDofsPerCell);
        mWorksetBase.worksetLocalState(aLocalState, tLocalStateWS);

        // Workset previous local state
        using PrevLocalStateScalar = typename JacobianPgrad::PrevLocalStateScalarType;
        Plato::ScalarMultiVectorT<PrevLocalStateScalar>
            tPrevLocalStateWS("Previous Local State Workset", mNumCells, mNumLocalDofsPerCell);
        mWorksetBase.worksetLocalState(aPrevLocalState, tPrevLocalStateWS);

        // Workset node state, i.e. projected pressure gradient
        using NodeStateScalar = typename JacobianPgrad::NodeStateScalarType;
        Plato::ScalarMultiVectorT<NodeStateScalar> tNodeStateWS("Node State Workset", mNumCells, mNumNodeStatePerCell);
        mWorksetBase.worksetNodeState(aNodeState, tNodeStateWS);

        // Workset control
        using ControlScalar = typename JacobianPgrad::ControlScalarType;
        Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", mNumCells, mNumNodesPerCell);
        mWorksetBase.worksetControl(aControl, tControlWS);

        // create return view
        using JacobianScalar = typename JacobianPgrad::ResultScalarType;
        Plato::ScalarMultiVectorT<JacobianScalar> tJacobianWS("Jacobian Node State Workset", mNumCells, mNumGlobalDofsPerCell);

        // Call evaluate function - compute Jacobian wrt pressure gradient
        mGlobalVecFuncJacPgrad->evaluate(tGlobalStateWS, tPrevGlobalStateWS, tLocalStateWS, tPrevLocalStateWS,
                                         tNodeStateWS, tControlWS, tConfigWS, tJacobianWS, aTimeStep);

        auto tOutput = this->assembleJacobianPressGrad(tJacobianWS);

        return (tOutput);
    }

    /***********************************************************************//**
     * \brief Compute assembled transpose Jacobian wrt pressure gradient of the global residual
     * \param [in] aGlobalState global state at current time step
     * \param [in] aPrevGlobalState global state at previous time step
     * \param [in] aLocalState local state at current time step
     * \param [in] aPrevLocalState local state at previous time step
     * \param [in] aControl control parameters
     * \param [in] aNodeState pressure gradient
     * \param [in] aTimeStep time step
     * \return Assembled transpose Jacobian wrt pressure gradient of the global residual
    ***************************************************************************/
    Teuchos::RCP<Plato::CrsMatrixType>
    gradient_n_T(const Plato::ScalarVector & aGlobalState,
                 const Plato::ScalarVector & aPrevGlobalState,
                 const Plato::ScalarVector & aLocalState,
                 const Plato::ScalarVector & aPrevLocalState,
                 const Plato::ScalarVector & aNodeState,
                 const Plato::ScalarVector & aControl,
                 Plato::Scalar aTimeStep = 0.0) const
    {
        // Workset config
        using ConfigScalar = typename JacobianPgrad::ConfigScalarType;
        Plato::ScalarArray3DT<ConfigScalar>
            tConfigWS("Config Workset", mNumCells, mNumNodesPerCell, mNumSpatialDims);
        mWorksetBase.worksetConfig(tConfigWS);

        // Workset current global state
        using GlobalStateScalar = typename JacobianPgrad::StateScalarType;
        Plato::ScalarMultiVectorT<GlobalStateScalar>
            tGlobalStateWS("Current Global State Workset", mNumCells, mNumGlobalDofsPerCell);
        mWorksetBase.worksetState(aGlobalState, tGlobalStateWS);

        // Workset previous global state
        using PrevGlobalStateScalar = typename JacobianPgrad::PrevStateScalarType;
        Plato::ScalarMultiVectorT<PrevGlobalStateScalar>
            tPrevGlobalStateWS("Previous Global State Workset", mNumCells, mNumGlobalDofsPerCell);
        mWorksetBase.worksetState(aPrevGlobalState, tPrevGlobalStateWS);

        // Workset local state
        using LocalStateScalar = typename JacobianPgrad::LocalStateScalarType;
        Plato::ScalarMultiVectorT<LocalStateScalar>
            tLocalStateWS("Local Current State Workset", mNumCells, mNumLocalDofsPerCell);
        mWorksetBase.worksetLocalState(aLocalState, tLocalStateWS);

        // Workset previous local state
        using PrevLocalStateScalar = typename JacobianPgrad::PrevLocalStateScalarType;
        Plato::ScalarMultiVectorT<PrevLocalStateScalar>
            tPrevLocalStateWS("Previous Local State Workset", mNumCells, mNumLocalDofsPerCell);
        mWorksetBase.worksetLocalState(aPrevLocalState, tPrevLocalStateWS);

        // Workset node state, i.e. projected pressure gradient
        using NodeStateScalar = typename JacobianPgrad::NodeStateScalarType;
        Plato::ScalarMultiVectorT<NodeStateScalar> tNodeStateWS("Node State Workset", mNumCells, mNumNodeStatePerCell);
        mWorksetBase.worksetNodeState(aNodeState, tNodeStateWS);

        // Workset control
        using ControlScalar = typename JacobianPgrad::ControlScalarType;
        Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", mNumCells, mNumNodesPerCell);
        mWorksetBase.worksetControl(aControl, tControlWS);

        // create return view
        using JacobianScalar = typename JacobianPgrad::ResultScalarType;
        Plato::ScalarMultiVectorT<JacobianScalar> tJacobianWS("Jacobian Node State Workset", mNumCells, mNumGlobalDofsPerCell);

        // Call evaluate function - compute Jacobian wrt pressure gradient
        mGlobalVecFuncJacPgrad->evaluate(tGlobalStateWS, tPrevGlobalStateWS, tLocalStateWS, tPrevLocalStateWS,
                                         tNodeStateWS, tControlWS, tConfigWS, tJacobianWS, aTimeStep);

        auto tOutput = this->assembleTransposeJacobianPressGrad(tJacobianWS);

        return (tOutput);
    }

// Private access functions
private:
    /***********************************************************************//**
     * \brief Assemble Jacobian for the projected pressure gradient problem
     *
     * \tparam AViewType POD type for 2-D Kokkos::View
     *
     * \param [in] aJacobianWS Jacobian worset
     * \return Assembled Jacobian
    ***************************************************************************/
    template<typename AViewType>
    Teuchos::RCP<Plato::CrsMatrixType>
    assembleJacobianPressGrad(const Plato::ScalarMultiVectorT<AViewType>& aJacobianWS)
    {
        // tJacobian has shape (Nc, (Nv x Nd), (Nv x Nn))
        //   Nc: number of cells
        //   Nv: number of vertices per cell
        //   Nd: dimensionality of the vector function
        //   Nn: dimensionality of the node state
        //   (I x J) is a strided 1D array indexed by i*J+j where i /in I and j /in J.
        //
        // (tJacobian is (Nc, (Nv x Nd)) and the third dimension, (Nv x Nn), is in the AD type)

        // create matrix with block size (Nd, Nn).
        //
        auto tMesh = mGlobalVecFuncJacPgrad->getMesh();
        Teuchos::RCP<Plato::CrsMatrixType> tAssembledJacobian =
                Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumSpatialDims, mNumNodeStatePerNode>( &tMesh );

        // create entry ordinal functor:
        // tJacobianMatEntryOrdinal(e, k, l) => G
        //   e: cell index
        //   k: row index /in (Nv x Nd)
        //   l: col index /in (Nv x Nn)
        //   G: entry index into CRS matrix
        //
        // Template parameters:
        //   mNumSpatialDims: Nv-1
        //   mNumSpatialDims: Nd
        //   mNumNodeStatePerNode:   Nn
        //
        // Note that the second two template parameters must match the block shape of the destination matrix, tJacobianMat
        //
        Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumSpatialDims, mNumNodeStatePerNode>
            tJacobianMatEntryOrdinal( tAssembledJacobian, &tMesh );

        // Assemble from the AD-typed result, tJacobian, into the POD-typed global matrix, tJacobianMat.
        //
        // Arguments 1 and 2 below correspond to the size of tJacobian ((Nv x Nd), (Nv x Nn)) and the size of
        // tJacobianMat (Nd, Nn).
        //
        auto tJacobianMatEntries = tAssembledJacobian->entries();
        mWorksetBase.assembleJacobian(
           mNumGlobalDofsPerCell,     // (Nv x Nd)
           mNumNodeStatePerCell,      // (Nv x Nn)
           tJacobianMatEntryOrdinal,  // entry ordinal functor
           aJacobianWS,               // source data
           tJacobianMatEntries        // destination
        );

        return tAssembledJacobian;
    }

    /***********************************************************************//**
     * \brief Assemble transpose Jacobian for the projected pressure gradient problem
     *
     * \tparam AViewType POD type for 2-D Kokkos::View
     *
     * \param [in] aJacobianWS Jacobian worset
     * \return Assembled transpose Jacobian
    ***************************************************************************/
    template<typename AViewType>
    Teuchos::RCP<Plato::CrsMatrixType>
    assembleTransposeJacobianPressGrad(const Plato::ScalarMultiVectorT<AViewType>& aJacobianWS)
    {
        // tJacobian has shape (Nc, (Nv x Nd), (Nv x Nn))
        //   Nc: number of cells
        //   Nv: number of vertices per cell
        //   Nd: dimensionality of the vector function
        //   Nn: dimensionality of the node state
        //   (I x J) is a strided 1D array indexed by i*J+j where i /in I and j /in J.
        //
        // (tJacobian is (Nc, (Nv x Nd)) and the third dimension, (Nv x Nn), is in the AD type)

        // create *transpose* matrix with block size (Nn, Nd).
        //
        auto tMesh = mGlobalVecFuncJacPgrad->getMesh();
        Teuchos::RCP<Plato::CrsMatrixType> tAssembledTransposeJacobian =
                Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumNodeStatePerNode, mNumGlobalDofsPerCell>( &tMesh );

        // create entry ordinal functor:
        // tJacobianMatEntryOrdinal(e, k, l) => G
        //   e: cell index
        //   k: row index /in (Nv x Nd)
        //   l: col index /in (Nv x Nn)
        //   G: entry index into CRS matrix
        //
        // Template parameters:
        //   mNumSpatialDims: Nv-1
        //   mNumNodeStatePerNode:   Nn
        //   mNumDofsPerNode: Nd
        //
        // Note that the second two template parameters must match the block shape of the destination matrix, tJacobianMat
        //
        Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumNodeStatePerNode, mNumGlobalDofsPerCell>
            tJacobianMatEntryOrdinal( tAssembledTransposeJacobian, &tMesh );

        // Assemble from the AD-typed result, tJacobian, into the POD-typed global matrix, tJacobianMat.
        //
        // The transpose is being assembled, (i.e., tJacobian is transposed before assembly into tJacobianMat), so
        // arguments 1 and 2 below correspond to the size of tJacobian ((Nv x Nd), (Nv x Nn)) and the size of the
        // *transpose* of tJacobianMat (Transpose(Nn, Nd) => (Nd, Nn)).
        //
        auto tJacobianMatEntries = tAssembledTransposeJacobian->entries();
        mWorksetBase.assembleTransposeJacobian(
           mNumGlobalDofsPerCell,     // (Nv x Nd)
           mNumNodeStatePerCell,      // (Nv x Nn)
           tJacobianMatEntryOrdinal,  // entry ordinal functor
           aJacobianWS,               // source data
           tJacobianMatEntries        // destination
        );

        return tAssembledTransposeJacobian;
    }
};
// class VectorFunctionVMSInc








/***************************************************************************//**
 * \brief Abstract interface for scalar function with local history variables
*******************************************************************************/
class ScalarFunctionLocalHistBase
{
public:
    /***************************************************************************//**
     * \brief Destructor
    *******************************************************************************/
    virtual ~ScalarFunctionLocalHistBase(){}

    /***************************************************************************//**
     * \brief Return function name
     * \return user defined function name
    *******************************************************************************/
    virtual std::string name() const = 0;

    /***************************************************************************//**
     * \brief Return function value
     * \param [in] aGlobalStates 2D view of global states
     * \param [in] aLocalStates 2D view of local states
     * \param [in] aControl 1D view of controls, i.e. design variables
     * \param [in] aTimeStep current time step increment
     * \return function value
    *******************************************************************************/
    virtual Plato::Scalar value(const Plato::ScalarVector & aGlobalStates,
                                const Plato::ScalarVector & aLocalStates,
                                const Plato::ScalarVector & aControl,
                                Plato::Scalar aTimeStep = 0.0) const = 0;

    /***************************************************************************//**
     * \brief Return partial derivative wrt design variables
     * \param [in] aGlobalStates 2D view of global states
     * \param [in] aLocalStates 2D view of local states
     * \param [in] aControl 1D view of controls, i.e. design variables
     * \param [in] aTimeStep current time step increment
     * \return assembled partial derivative wrt design variables
    *******************************************************************************/
    virtual Plato::ScalarVector gradient_z(const Plato::ScalarVector & aGlobalStates,
                                           const Plato::ScalarVector & aLocalStates,
                                           const Plato::ScalarVector & aControl,
                                           Plato::Scalar aTimeStep = 0.0) const = 0;

    /***************************************************************************//**
     * \brief Return partial derivative wrt global states workset
     * \param [in] aGlobalStates 2D view of global states
     * \param [in] aLocalStates 2D view of local states
     * \param [in] aControl 1D view of controls, i.e. design variables
     * \param [in] aTimeStep current time step increment
     * \return partial derivative wrt global states workset
    *******************************************************************************/
    virtual Plato::ScalarMultiVector gradient_u(const Plato::ScalarVector & aGlobalStates,
                                                const Plato::ScalarVector & aLocalStates,
                                                const Plato::ScalarVector & aControl,
                                                Plato::Scalar aTimeStep = 0.0) const = 0;

    /***************************************************************************//**
     * \brief Return partial derivative wrt local states workset
     * \param [in] aGlobalStates 2D view of global states
     * \param [in] aLocalStates 2D view of local states
     * \param [in] aControl 1D view of controls, i.e. design variables
     * \param [in] aTimeStep current time step increment
     * \return partial derivative wrt local states workset
    *******************************************************************************/
    virtual Plato::ScalarMultiVector gradient_c(const Plato::ScalarVector & aGlobalStates,
                                                const Plato::ScalarVector & aLocalStates,
                                                const Plato::ScalarVector & aControl,
                                                Plato::Scalar aTimeStep = 0.0) const = 0;

    /***************************************************************************//**
     * \brief Return assembled partial derivative wrt configurtion variables
     * \param [in] aGlobalStates 2D view of global states
     * \param [in] aLocalStates 2D view of local states
     * \param [in] aControl 1D view of controls, i.e. design variables
     * \param [in] aTimeStep current time step increment
     * \return assembled partial derivative wrt configurtion variables
    *******************************************************************************/
    virtual Plato::ScalarVector gradient_x(const Plato::ScalarVector & aGlobalStates,
                                           const Plato::ScalarVector & aLocalStates,
                                           const Plato::ScalarVector & aControl,
                                           Plato::Scalar aTimeStep = 0.0) const = 0;

    /***************************************************************************//**
     * \brief Update physics-based parameters within optimization iterations
     * \param [in] aGlobalStates 2D view of global states
     * \param [in] aLocalStates 2D view of local states
     * \param [in] aControl 1D view of controls, i.e. design variables
    *******************************************************************************/
    virtual void updateProblem(const Plato::ScalarMultiVector & aGlobalStates,
                               const Plato::ScalarMultiVector & aLocalStates,
                               const Plato::ScalarVector & aControl) const = 0;
};
// class ScalarFunctionVMSIncBase






/***************************************************************************//**
 * \brief Data structure used in Plasticity Problem.  The Plasticity Problem
 * interface is responsible of evaluating the system of forward and adjoint
 * equations as well as assembling the total gradient with respect to the
 * variables of interest, e.g. design variables, configurations, etc.
*******************************************************************************/
struct StateData
{
    Plato::Scalar mTimeStep;               /*!< current time step */
    Plato::OrdinalType mCurrentStepIndex;  /*!< current time step index */
    Plato::OrdinalType mPreviousStepIndex; /*!< previous time step index */

    Plato::ScalarVector mCurrentLocalState;    /*!< current local state */
    Plato::ScalarVector mPreviousLocalState;   /*!< previous local state */
    Plato::ScalarVector mCurrentGlobalState;   /*!< current global state */
    Plato::ScalarVector mPreviousGlobalState;  /*!< previous global state */
    Plato::ScalarVector mCurrentProjPressGrad; /*!< current projected pressure gradient */
};
// struct StateData

struct AdjointData
{
    Plato::ScalarVector mCurrentLocalAdjoint;   /*!< current local adjoint */
    Plato::ScalarVector mPreviousLocalAdjoint;  /*!< previous local adjoint */
    Plato::ScalarVector mCurrentGlobalAdjoint;  /*!< current global adjoint */
    Plato::ScalarVector mPreviousGlobalAdjoint; /*!< previous global adjoint */
};
// struct AdjointData











/*
*****************************************************************************
 * \brief Plasticity problem manager, which is responsible for performance
 * criteria evaluations and
 * \param [in] aMesh mesh database
 * \param [in] aMeshSets side sets database
 * \param [in] aInputParams input parameters database
*********************************************************************************
template<typename SimplexPhysics>
class PlasticityProblem : public Plato::AbstractProblem
{
// private member data
private:
    static constexpr auto mSpatialDim = SimplexPhysics::mNumSpatialDims;               !< spatial dimensions
    static constexpr auto mNumNodesPerCell = SimplexPhysics::mNumNodesPerCell;         !< number of nodes per cell
    static constexpr auto mPressureDofOffset = SimplexPhysics::mPressureDofOffset;     !< number of pressure dofs offset
    static constexpr auto mNumGlobalDofsPerNode = SimplexPhysics::mNumDofsPerNode;     !< number of global degrees of freedom per node
    static constexpr auto mNumGlobalDofsPerCell = SimplexPhysics::mNumDofsPerCell;     !< number of global degrees of freedom per cell (i.e. element)
    static constexpr auto mNumLocalDofsPerCell = SimplexPhysics::mNumLocalDofsPerCell; !< number of local degrees of freedom per cell (i.e. element)

    // Required
    Plato::VectorFunctionVMSInc<SimplexPhysics> mGlobalResidualEq;      !< global equality constraint interface
    Plato::LocalVectorFunctionInc<SimplexPhysics> mLocalResidualEq;     !< local equality constraint interface
    Plato::VectorFunctionVMS<SimplexPhysics::ProjectorT> mProjectionEq; !< global pressure gradient projection interface

    // Optional
    std::shared_ptr<Plato::ScalarFunctionLocalHistBase> mObjective;  !< objective constraint interface
    std::shared_ptr<Plato::ScalarFunctionLocalHistBase> mConstraint; !< constraint constraint interface

    Plato::Scalar mPseudoTimeStep;             !< pseudo time step increment
    Plato::Scalar mInitialNormResidual;        !< initial norm of global residual
    Plato::Scalar mCurrentPseudoTimeStep;      !< current pseudo time step
    Plato::Scalar mNewtonRaphsonStopTolerance; !< Newton-Raphson stopping tolerance

    Plato::OrdinalType mNumPseudoTimeSteps;    !< maximum number of pseudo time steps
    Plato::OrdinalType mMaxNumNewtonIter;      !< maximum number of Newton-Raphson iterations

    Plato::ScalarVector mGlobalResidual;       !< global residual
    Plato::ScalarVector mProjResidual;         !< projection residual, i.e. projected pressure gradient solve residual
    Plato::ScalarVector mProjectedPressure;    !< projected pressure
    Plato::ScalarVector mProjectionAdjoint;    !< projection adjoint variables

    Plato::ScalarMultiVector mLocalStates;        !< local state variables
    Plato::ScalarMultiVector mLocalAdjoint;       !< local adjoint variables
    Plato::ScalarMultiVector mGlobalStates;       !< global state variables
    Plato::ScalarMultiVector mGlobalAdjoint;      !< global adjoint variables
    Plato::ScalarMultiVector mProjectedPressGrad; !< projected pressure gradient (# Time Steps, # Projected Pressure Gradient dofs)

    Teuchos::RCP<Plato::CrsMatrixType> mProjectionJacobian;   !< projection residual Jacobian matrix
    Teuchos::RCP<Plato::CrsMatrixType> mGlobalJacobian; !< global residual Jacobian matrix

    Plato::ScalarVector mDirichletValues; !< values associated with the Dirichlet boundary conditions
    Plato::ScalarVector mDispControlDirichletValues; !< values associated with the Dirichlet boundary conditions at the current pseudo time step
    Plato::LocalOrdinalVector mDirichletDofs; !< list of degrees of freedom associated with the Dirichlet boundary conditions

    Plato::WorksetBase<SimplexPhysics> mWorksetBase; !< assembly routine interface
    std::shared_ptr<Plato::BlockMatrixEntryOrdinal<mSpatialDim, mNumGlobalDofsPerNode>> mGlobalJacEntryOrdinal; !< global Jacobian matrix entry ordinal

// public functions
public:
    *****************************************************************************
     * \brief PLATO Plasticity Problem constructor
     * \param [in] aMesh mesh database
     * \param [in] aMeshSets side sets database
     * \param [in] aInputParams input parameters database
    *********************************************************************************
    PlasticityProblem(Omega_h::Mesh& aMesh, Omega_h::MeshSets& aMeshSets, Teuchos::ParameterList& aInputParams) :
            mGlobalResidualEq(aMesh, aMeshSets, mDataMap, aInputParams, aInputParams.get<std::string>("PDE Constraint")),
            mLocalResidualEq(aMesh, aMeshSets, mDataMap, aInputParams, aInputParams.get<std::string>("Plasticity Model")),
            mProjectionEq(aMesh, aMeshSets, mDataMap, aInputParams, std::string("State Gradient Projection")),
            mObjective(nullptr),
            mConstraint(nullptr),
            mPseudoTimeStep(Plato::ParseTools::getSubParam<Plato::Scalar>(aInputParams, "Time Stepping", "Time Step", 1.0)),
            mCurrentPseudoTimeStep(0.0),
            mInitialNormResidual(std::numeric_limits<Plato::Scalar>::max()),
            mNewtonRaphsonStopTolerance(Plato::ParseTools::getSubParam<Plato::Scalar>(aInputParams, "Newton-Raphson", "Stopping Tolerance", 1e-8)),
            mNumPseudoTimeSteps(Plato::ParseTools::getSubParam<Plato::OrdinalType>(aInputParams, "Time Stepping", "Number Time Steps", 2)),
            mMaxNumNewtonIter(Plato::ParseTools::getSubParam<Plato::OrdinalType>(aInputParams, "Newton-Raphson", "Number Iterations", 10)),
            mGlobalResidual("Global Residual", mGlobalResidualEq.size()),
            mProjResidual("Projected Residual", mProjectionEq.size()),
            mProjectedPressure("Project Pressure", aMesh.nverts()),
            mLocalStates("Local States", mNumPseudoTimeSteps, mLocalResidualEq.size()),
            mGlobalStates("Global States", mNumPseudoTimeSteps, mGlobalResidualEq.size()),
            mProjectedPressGrad("Projected Pressure Gradient", mNumPseudoTimeSteps, mProjectionEq.size())
    {
        this->initialize(aMesh, aMeshSets, aInputParams);
    }

    *****************************************************************************
     * \brief PLATO Plasticity Problem destructor
    *********************************************************************************
    virtual ~PlasticityProblem(){}

    *****************************************************************************
     * \brief Return number of global degrees of freedom in solution.
     * \return Number of global degrees of freedom
    *********************************************************************************
    Plato::OrdinalType getNumSolutionDofs()
    {
        return (mGlobalResidualEq.size());
    }

    *****************************************************************************
     * \brief Set global state variables
     * \param [in] aState 2D view of global state variables - (NumTimeSteps, TotalDofs)
    *********************************************************************************
    void setState(const Plato::ScalarMultiVector & aState)
    {
        assert(aState.extent(0) == mGlobalStates.extent(0));
        assert(aState.extent(1) == mGlobalStates.extent(1));
        Kokkos::deep_copy(mGlobalStates, aState);
    }

    *****************************************************************************
     * \brief Return 2D view of global state variables - (NumTimeSteps, TotalDofs)
     * \return aState 2D view of global state variables
    *********************************************************************************
    Plato::ScalarMultiVector getState()
    {
        return mGlobalStates;
    }

    *****************************************************************************
     * \brief Return 2D view of global adjoint variables - (2, TotalDofs)
     * \return 2D view of global adjoint variables
    *********************************************************************************
    Plato::ScalarMultiVector getAdjoint()
    {
        return mGlobalAdjoint;
    }

    *****************************************************************************
     * \brief Apply Dirichlet constraints
     * \param [in] aMatrix Compressed Row Storage (CRS) matrix
     * \param [in] aVector 1D view of Right-Hand-Side forces
    *********************************************************************************
    void applyConstraints(const Teuchos::RCP<Plato::CrsMatrixType> & aMatrix, const Plato::ScalarVector & aVector)
    {
        // apply displacement control, i.e. continuation
        Plato::update(mCurrentPseudoTimeStep, mDirichletValues, static_cast<Plato::Scalar>(0), mDispControlDirichletValues);

        if(mGlobalJacobian->isBlockMatrix())
        {
            Plato::applyBlockConstraints<mNumGlobalDofsPerNode>(aMatrix, aVector, mDirichletDofs, mDispControlDirichletValues);
        }
        else
        {
            Plato::applyConstraints<mNumGlobalDofsPerNode>(aMatrix, aVector, mDirichletDofs, mDispControlDirichletValues);
        }
    }

    *****************************************************************************
     * \brief Fill right-hand-side vector values
    *********************************************************************************
    void applyBoundaryLoads(const Plato::ScalarVector & aForce) { return; }

    *****************************************************************************
     * \brief Update physics-based parameters within optimization iterations
     * \param [in] aControl 1D container of control variables
     * \param [in] aGlobalState 2D container of global state variables
    *********************************************************************************
    void updateProblem(const Plato::ScalarVector & aControl, const Plato::ScalarMultiVector & aGlobalState)
    {
        mObjective->updateProblem(aGlobalState, mLocalStates, aControl);
        mConstraint->updateProblem(aGlobalState, mLocalStates, aControl);
    }

    *****************************************************************************
     * \brief Solve system of equations
     * \param [in] aControl 1D view of control variables
     * \return 2D view of state variables
    *********************************************************************************
    Plato::ScalarMultiVector solution(const Plato::ScalarVector & aControl)
    {
        Plato::StateData tStateData;
        auto tNumCells = mLocalResidualEq.numCells();
        Plato::ScalarArray3D tInvLocalJacobianT("Inverse Transpose DhDc", tNumCells, mNumLocalDofsPerCell, mNumLocalDofsPerCell);

        // outer loop for load/time steps
        Plato::ScalarVector tGlobalStateIncrement("Global State increment", mGlobalResidualEq.size());
        // todo: make sure i go over all the time steps. right now, i only go up to n-1, where n is the total number of time steps
        for(Plato::OrdinalType tCurrentStepIndex = 1; tCurrentStepIndex < mNumPseudoTimeSteps; tCurrentStepIndex++)
        {
            tStateData.mTimeStep = 0;
            tStateData.mCurrentStepIndex = tCurrentStepIndex;
            tStateData.mPreviousStepIndex = tCurrentStepIndex - static_cast<Plato::OrdinalType>(1);
            this->updateStateData(tStateData, true  set entries to zero );

            // inner loop for load/time steps
            mCurrentPseudoTimeStep = static_cast<Plato::Scalar>(tCurrentStepIndex) * mPseudoTimeStep;
            for(Plato::OrdinalType tNewtonIteration = 0; tNewtonIteration < mMaxNumNewtonIter; tNewtonIteration++)
            {
                // compute projected pressure gradient
                mProjResidual = mProjectionEq.value(tStateData.mCurrentProjPressGrad, mProjectedPressure, aControl);
                mProjectionJacobian = mProjectionEq.gradient_u(tStateData.mCurrentProjPressGrad, mProjectedPressure, aControl);
                Plato::Solve::RowSummed<SimplexPhysics::mNumSpatialDims>(mProjectionJacobian, tStateData.mCurrentProjPressGrad, mProjResidual);

                // compute the global state residual
                mGlobalResidual = mGlobalResidualEq.value(tStateData.mCurrentGlobalState, tStateData.mPreviousGlobalState,
                                                          tStateData.mCurrentLocalState, tStateData.mPreviousLocalState,
                                                          tStateData.mCurrentProjPressGrad, aControl);

                // check convergence
                if(this->checkNewtonRaphsonStoppingCriteria(tNewtonIteration) == true)
                {
                    break;
                }

                // update inverse of local jacobian -> store in tInvLocalJacobianT
                this->updateInverseLocalJacobian(aControl, tStateData, tInvLocalJacobianT);
                // assemble tangent stiffness matrix
                this->assembleTangentStiffnessMatrix(aControl, tStateData, tInvLocalJacobianT);
                // apply dirichlet conditions
                this->applyConstraints(mGlobalJacobian, mGlobalResidual);
                // solve global system of equations
                Plato::fill(static_cast<Plato::Scalar>(0.0), tGlobalStateIncrement);
                Plato::Solve::Consistent<mNumGlobalDofsPerNode>(mGlobalJacobian, tGlobalStateIncrement, mGlobalResidual);
                // update global state
                this->zeroDirichletDofs(tGlobalStateIncrement);
                Plato::update(static_cast<Plato::Scalar>(-1.0), tGlobalStateIncrement, static_cast<Plato::Scalar>(1.0), tStateData.mCurrentGlobalState);
                // update local state
                mLocalResidualEq.updateLocalState(tStateData.mCurrentGlobalState, tStateData.mPreviousGlobalState,
                                                  tStateData.mCurrentLocalState, tStateData.mPreviousLocalState, aControl);
                // copy projection state, i.e. pressure
                Plato::extract<mNumGlobalDofsPerNode, mPressureDofOffset>(tStateData.mCurrentGlobalState, mProjectedPressure);
            }
        }

        return mGlobalStates;
    }

    *****************************************************************************
     * \fn Plato::Scalar objectiveValue(const Plato::ScalarVector & aControl,
     *                                  const Plato::ScalarMultiVector & aGlobalState)
     * \brief Evaluate objective function and return its value
     * \param [in] aControl 1D view of control variables
     * \param [in] aGlobalState 2D view of state variables
     * \return objective function value
    *********************************************************************************
    Plato::Scalar objectiveValue(const Plato::ScalarVector & aControl, const Plato::ScalarMultiVector & aGlobalState)
    {
        if(aControl.size() <= static_cast<Plato::OrdinalType>(0))
        {
            THROWERR("\nCONTROL 1D VIEW IS EMPTY.\n");
        }
        if(aGlobalState.size() <= static_cast<Plato::OrdinalType>(0))
        {
            THROWERR("\nGLOBAL STATE 2D VIEW IS EMPTY.\n");
        }
        if(mObjective == nullptr)
        {
            THROWERR("\nOBJECTIVE PTR IS NULL.\n");
        }

        return mObjective->value(aGlobalState, mLocalStates, aControl);
    }

    *****************************************************************************
     * \fn Plato::Scalar objectiveValue(const Plato::ScalarVector & aControl)
     * \brief Evaluate objective function and return its value
     * \param [in] aControl 1D view of control variables
     * \return objective function value
    *********************************************************************************
    Plato::Scalar objectiveValue(const Plato::ScalarVector & aControl)
    {
        if(aControl.size() <= static_cast<Plato::OrdinalType>(0))
        {
            THROWERR("\nCONTROL 1D VIEW IS EMPTY.\n");
        }
        if(mObjective == nullptr)
        {
            THROWERR("\nOBJECTIVE PTR IS NULL.\n");
        }

        auto tGlobalStates = this->solution(aControl);
        return mObjective->value(tGlobalStates, mLocalStates, aControl);
    }

    *****************************************************************************
     * \fn Plato::Scalar constraintValue(const Plato::ScalarVector & aControl, const Plato::ScalarMultiVector & aState)
     * \brief Evaluate constraint function and return its value
     * \param [in] aControl 1D view of control variables
     * \param [in] aGlobalState 2D view of state variables
     * \return constraint function value
    *********************************************************************************
    Plato::Scalar constraintValue(const Plato::ScalarVector & aControl, const Plato::ScalarMultiVector & aGlobalState)
    {
        if(aControl.size() <= static_cast<Plato::OrdinalType>(0))
        {
            THROWERR("\nCONTROL 1D VIEW IS EMPTY.\n");
        }
        if(aGlobalState.size() <= static_cast<Plato::OrdinalType>(0))
        {
            THROWERR("\nGLOBAL STATE 2D VIEW IS EMPTY.\n");
        }
        if(mConstraint == nullptr)
        {
            THROWERR("\nCONSTRAINT PTR IS NULL.\n");
        }

        return mConstraint->value(aGlobalState, mLocalStates, aControl);
    }

    *****************************************************************************
     * \fn Plato::Scalar constraintValue(const Plato::ScalarVector & aControl)
     * \brief Evaluate constraint function and return its value
     * \param [in] aControl 1D view of control variables
     * \return constraint function value
    *********************************************************************************
    Plato::Scalar constraintValue(const Plato::ScalarVector & aControl)
    {
        if(aControl.size() <= static_cast<Plato::OrdinalType>(0))
        {
            THROWERR("\nCONTROL 1D VIEW IS EMPTY.\n");
        }
        if(mConstraint == nullptr)
        {
            THROWERR("\nCONSTRAINT PTR IS NULL.\n");
        }

        return mConstraint->value(mGlobalStates, mLocalStates, aControl);
    }

    *****************************************************************************
     * \brief Evaluate objective partial derivative wrt control variables
     * \param [in] aControl 1D view of control variables
     * \return 1D view - objective partial derivative wrt control variables
    *********************************************************************************
    Plato::ScalarVector objectiveGradient(const Plato::ScalarVector & aControl)
    {
        if(aControl.size() <= static_cast<Plato::OrdinalType>(0))
        {
            THROWERR("\nCONTROL 1D VIEW IS EMPTY.\n");
        }
        if(mObjective == nullptr)
        {
            THROWERR("\nOBJECTIVE PTR IS NULL.\n");
        }

        return mObjective->gradient_z(mGlobalStates, mLocalStates, aControl);
    }

    *****************************************************************************
     * \brief Evaluate objective gradient wrt control variables
     * \param [in] aControl 1D view of control variables
     * \param [in] aGlobalState 2D view of global state variables
     * \return 1D view of the objective gradient wrt control variables
    *********************************************************************************
    Plato::ScalarVector objectiveGradient(const Plato::ScalarVector & aControl, const Plato::ScalarMultiVector & aGlobalState)
    {
        if(aControl.size() <= static_cast<Plato::OrdinalType>(0))
        {
            THROWERR("\nCONTROL 1D VIEW IS EMPTY.\n");
        }
        if(aGlobalState.size() <= static_cast<Plato::OrdinalType>(0))
        {
            THROWERR("\nGLOBAL STATE 2D VIEW IS EMPTY.\n");
        }
        if(mObjective == nullptr)
        {
            THROWERR("\nOBJECTIVE PTR IS NULL.\n");
        }

        // Create state data manager
        Plato::StateData tStateData;
        Plato::AdjointData tAdjointData;
        auto tNumCells = mLocalResidualEq.numCells();
        Plato::ScalarArray3D tInvLocalJacobianT("Inverse Transpose DhDc", tNumCells, mNumLocalDofsPerCell, mNumLocalDofsPerCell);

        // Compute Partial Derivative of Criterion wrt Controls
        auto tDfDz = mObjective->gradient_z(aGlobalState, mLocalStates, aControl);

        // outer loop for pseudo time steps
        auto tLastStepIndex = mNumPseudoTimeSteps - static_cast<Plato::OrdinalType>(1);
        for(Plato::OrdinalType tCurrentStepIndex = tLastStepIndex; tCurrentStepIndex > 0; tCurrentStepIndex--)
        {
            tStateData.mTimeStep = 0;
            tStateData.mCurrentStepIndex = tCurrentStepIndex;
            tStateData.mPreviousStepIndex = tCurrentStepIndex + static_cast<Plato::OrdinalType>(1);

            this->updateStateData(tStateData);
            this->updateAdjointData(tAdjointData);
            this->updateInverseLocalJacobian(aControl, tStateData, tInvLocalJacobianT);

            this->updateGlobalAdjoint(aControl, tStateData, tInvLocalJacobianT, tAdjointData, *mObjective);
            this->updateLocalAdjoint(aControl, tStateData, tInvLocalJacobianT, tAdjointData, *mObjective);
            this->updateProjectionAdjoint(aControl, tStateData, tAdjointData);

            this->updateGradientControl(aControl, tStateData, tAdjointData, tDfDz);
        }

        return (tDfDz);
    }

    *****************************************************************************
     * \brief Evaluate objective partial derivative wrt configuration variables
     * \param [in] aControl 1D view of control variables
     * \return 1D view - objective partial derivative wrt configuration variables
    *********************************************************************************
    Plato::ScalarVector objectiveGradientX(const Plato::ScalarVector & aControl)
    {
        if(aControl.size() <= static_cast<Plato::OrdinalType>(0))
        {
            THROWERR("\nCONTROL 1D VIEW IS EMPTY.\n");
        }
        if(mObjective == nullptr)
        {
            THROWERR("\nOBJECTIVE PTR IS NULL.\n");
        }

        return mObjective->gradient_x(mGlobalStates, mLocalStates, aControl);
    }

    *****************************************************************************
     * \brief Evaluate objective gradient wrt configuration variables
     * \param [in] aControl 1D view of control variables
     * \param [in] aGlobalState 2D view of global state variables
     * \return 1D view of the objective gradient wrt control variables
    *********************************************************************************
    Plato::ScalarVector objectiveGradientX(const Plato::ScalarVector & aControl, const Plato::ScalarMultiVector & aGlobalState)
    {
        if(aControl.size() <= static_cast<Plato::OrdinalType>(0))
        {
            THROWERR("\nCONTROL 1D VIEW IS EMPTY.\n");
        }
        if(aGlobalState.size() <= static_cast<Plato::OrdinalType>(0))
        {
            THROWERR("\nGLOBAL STATE 2D VIEW IS EMPTY.\n");
        }
        if(mObjective == nullptr)
        {
            THROWERR("\nOBJECTIVE PTR IS NULL.\n");
        }

        // Create state data manager
        Plato::StateData tStateData;
        Plato::AdjointData tAdjointData;
        auto tNumCells = mLocalResidualEq.numCells();
        Plato::ScalarArray3D tInvLocalJacobianT("Inverse Transpose DhDc", tNumCells, mNumLocalDofsPerCell, mNumLocalDofsPerCell);

        // Compute Partial Derivative of Criterion wrt Controls
        auto tDfDx = mObjective->gradient_x(aGlobalState, mLocalStates, aControl);

        // outer loop for pseudo time steps
        auto tLastStepIndex = mNumPseudoTimeSteps - static_cast<Plato::OrdinalType>(1);
        for(Plato::OrdinalType tCurrentStepIndex = tLastStepIndex; tCurrentStepIndex > 0; tCurrentStepIndex--)
        {
            tStateData.mTimeStep = 0;
            tStateData.mCurrentStepIndex = tCurrentStepIndex;
            tStateData.mPreviousStepIndex = tCurrentStepIndex + static_cast<Plato::OrdinalType>(1);

            this->updateStateData(tStateData);
            this->updateAdjointData(tAdjointData);
            this->updateInverseLocalJacobian(aControl, tStateData, tInvLocalJacobianT);

            this->updateGlobalAdjoint(aControl, tStateData, tInvLocalJacobianT, tAdjointData, *mObjective);
            this->updateLocalAdjoint(aControl, tStateData, tInvLocalJacobianT, tAdjointData, *mObjective);
            this->updateProjectionAdjoint(aControl, tStateData, tAdjointData);

            this->updateGradientConfiguration(aControl, tStateData, tAdjointData, tDfDx);
        }

        return (tDfDx);
    }

    *****************************************************************************
     * \brief Evaluate constraint partial derivative wrt control variables
     * \param [in] aControl 1D view of control variables
     * \return 1D view - constraint partial derivative wrt control variables
    *********************************************************************************
    Plato::ScalarVector constraintGradient(const Plato::ScalarVector & aControl)
    {
        if(aControl.size() <= static_cast<Plato::OrdinalType>(0))
        {
            THROWERR("\nCONTROL 1D VIEW IS EMPTY.\n");
        }
        if(mConstraint == nullptr)
        {
            THROWERR("\nCONSTRAINT PTR IS NULL.\n");
        }

        return mConstraint->gradient_z(mGlobalStates, mLocalStates, aControl);
    }

    *****************************************************************************
     * \brief Evaluate constraint partial derivative wrt control variables
     * \param [in] aControl 1D view of control variables
     * \param [in] aState 2D view of state variables
     * \return 1D view - constraint partial derivative wrt control variables
    *********************************************************************************
    Plato::ScalarVector constraintGradient(const Plato::ScalarVector & aControl, const Plato::ScalarMultiVector & aGlobalState)
    {
        if(aControl.size() <= static_cast<Plato::OrdinalType>(0))
        {
            THROWERR("\nCONTROL 1D VIEW IS EMPTY.\n");
        }
        if(aGlobalState.size() <= static_cast<Plato::OrdinalType>(0))
        {
            THROWERR("\nGLOBAL STATE 2D VIEW IS EMPTY.\n");
        }
        if(mConstraint == nullptr)
        {
            THROWERR("\nCONSTRAINT PTR IS NULL.\n");
        }

        // Create state data manager
        Plato::StateData tStateData;
        Plato::AdjointData tAdjointData;
        auto tNumCells = mLocalResidualEq.numCells();
        Plato::ScalarArray3D tInvLocalJacobianT("Inverse Transpose DhDc", tNumCells, mNumLocalDofsPerCell, mNumLocalDofsPerCell);

        // Compute Partial Derivative of Criterion wrt Controls
        auto tDgDz = mConstraint->gradient_z(aGlobalState, mLocalStates, aControl);

        // outer loop for pseudo time steps
        auto tLastStepIndex = mNumPseudoTimeSteps - static_cast<Plato::OrdinalType>(1);
        for(Plato::OrdinalType tCurrentStepIndex = tLastStepIndex; tCurrentStepIndex > 0; tCurrentStepIndex--)
        {
            tStateData.mTimeStep = 0;
            tStateData.mCurrentStepIndex = tCurrentStepIndex;
            tStateData.mPreviousStepIndex = tCurrentStepIndex + static_cast<Plato::OrdinalType>(1);

            this->updateStateData(tStateData);
            this->updateAdjointData(tAdjointData);
            this->updateInverseLocalJacobian(aControl, tStateData, tInvLocalJacobianT);

            this->updateGlobalAdjoint(aControl, tStateData, tInvLocalJacobianT, tAdjointData, *mConstraint);
            this->updateLocalAdjoint(aControl, tStateData, tInvLocalJacobianT, tAdjointData, *mConstraint);
            this->updateProjectionAdjoint(aControl, tStateData, tAdjointData);

            this->updateGradientControl(aControl, tStateData, tAdjointData, tDgDz);
        }

        return (tDgDz);
    }

    *****************************************************************************
     * \brief Evaluate constraint partial derivative wrt configuration variables
     * \param [in] aControl 1D view of control variables
     * \return 1D view - constraint partial derivative wrt configuration variables
    *********************************************************************************
    Plato::ScalarVector constraintGradientX(const Plato::ScalarVector & aControl)
    {
        if(aControl.size() <= static_cast<Plato::OrdinalType>(0))
        {
            THROWERR("\nCONTROL 1D VIEW IS EMPTY.\n");
        }
        if(mConstraint == nullptr)
        {
            THROWERR("\nCONSTRAINT PTR IS NULL.\n");
        }

        return mConstraint->gradient_x(mGlobalStates, mLocalStates, aControl);
    }

    *****************************************************************************
     * \brief Evaluate constraint partial derivative wrt configuration variables
     * \param [in] aControl 1D view of control variables
     * \param [in] aState 2D view of state variables
     * \return 1D view - constraint partial derivative wrt configuration variables
    *********************************************************************************
    Plato::ScalarVector constraintGradientX(const Plato::ScalarVector & aControl, const Plato::ScalarMultiVector & aGlobalState)
    {
        if(aControl.size() <= static_cast<Plato::OrdinalType>(0))
        {
            THROWERR("\nCONTROL 1D VIEW IS EMPTY.\n");
        }
        if(aGlobalState.size() <= static_cast<Plato::OrdinalType>(0))
        {
            THROWERR("\nGLOBAL STATE 2D VIEW IS EMPTY.\n");
        }
        if(mConstraint == nullptr)
        {
            THROWERR("\nCONSTRAINT PTR IS NULL.\n");
        }

        // Create state data manager
        Plato::StateData tStateData;
        Plato::AdjointData tAdjointData;
        auto tNumCells = mLocalResidualEq.numCells();
        Plato::ScalarArray3D tInvLocalJacobianT("Inverse Transpose DhDc", tNumCells, mNumLocalDofsPerCell, mNumLocalDofsPerCell);

        // Compute Partial Derivative of Criterion wrt Controls
        auto tDgDx = mConstraint->gradient_x(aGlobalState, mLocalStates, aControl);

        // outer loop for pseudo time steps
        auto tLastStepIndex = mNumPseudoTimeSteps - static_cast<Plato::OrdinalType>(1);
        for(Plato::OrdinalType tCurrentStepIndex = tLastStepIndex; tCurrentStepIndex > 0; tCurrentStepIndex--)
        {
            tStateData.mTimeStep = 0;
            tStateData.mCurrentStepIndex = tCurrentStepIndex;
            tStateData.mPreviousStepIndex = tCurrentStepIndex + static_cast<Plato::OrdinalType>(1);

            this->updateStateData(tStateData);
            this->updateAdjointData(tAdjointData);
            this->updateInverseLocalJacobian(aControl, tStateData, tInvLocalJacobianT);

            this->updateGlobalAdjoint(aControl, tStateData, tInvLocalJacobianT, tAdjointData, *mConstraint);
            this->updateLocalAdjoint(aControl, tStateData, tInvLocalJacobianT, tAdjointData, *mConstraint);
            this->updateProjectionAdjoint(aControl, tStateData, tAdjointData);

            this->updateGradientConfiguration(aControl, tStateData, tAdjointData, tDgDx);
        }

        return (tDgDx);
    }

    *****************************************************************************
     * \brief Update state data for time step n, i.e. current time step:
     * \param [in] aStateData state data manager
     * \param [in] aZeroEntries flag - zero all entries in current states (default = false)
    *********************************************************************************
    void updateStateData(Plato::StateData &aStateData, bool aZeroEntries = false)
    {
        aStateData.mCurrentLocalState = Kokkos::subview(mLocalStates, aStateData.mCurrentStepIndex, Kokkos::ALL());
        aStateData.mCurrentGlobalState = Kokkos::subview(mGlobalStates, aStateData.mCurrentStepIndex, Kokkos::ALL());
        aStateData.mCurrentProjPressGrad = Kokkos::subview(mProjectedPressGrad, aStateData.mCurrentStepIndex, Kokkos::ALL());

        if(aZeroEntries == true)
        {
            Plato::fill(static_cast<Plato::Scalar>(0.0), aStateData.mCurrentLocalState);
            Plato::fill(static_cast<Plato::Scalar>(0.0), aStateData.mCurrentGlobalState);
            Plato::fill(static_cast<Plato::Scalar>(0.0), aStateData.mCurrentProjPressGrad);
            Plato::fill(static_cast<Plato::Scalar>(0.0), mProjectedPressure);
        }

        if (aStateData.mPreviousStepIndex != mNumPseudoTimeSteps)
        {
            aStateData.mPreviousLocalState = Kokkos::subview(mLocalStates, aStateData.mPreviousStepIndex, Kokkos::ALL());
            aStateData.mPreviousGlobalState = Kokkos::subview(mGlobalStates, aStateData.mPreviousStepIndex, Kokkos::ALL());
        }
    }

    *****************************************************************************
     * \brief Update adjoint data for time step n, i.e. current time step:
     * \param [in] aAdjointData adjoint data manager
    *********************************************************************************
    void updateAdjointData(Plato::AdjointData& aAdjointData)
    {
        const Plato::OrdinalType tCurrentStepIndex = 1;
        aAdjointData.mCurrentLocalAdjoint = Kokkos::subview(mLocalAdjoint, tCurrentStepIndex, Kokkos::ALL());
        aAdjointData.mCurrentGlobalAdjoint = Kokkos::subview(mGlobalAdjoint, tCurrentStepIndex, Kokkos::ALL());
        Plato::Scalar tAlpha = 1.0; Plato::Scalar tBeta = 0.0;
        Plato::update(tAlpha, aAdjointData.mCurrentLocalAdjoint, tBeta, aAdjointData.mPreviousLocalAdjoint);
        Plato::update(tAlpha, aAdjointData.mCurrentGlobalAdjoint, tBeta, aAdjointData.mPreviousGlobalAdjoint);
    }

    *****************************************************************************
     * \brief Update inverse of local Jacobian wrt local states, i.e.
     * /f$ \left[ \left( \frac{\partial{H}}{\partial{c}} \right)_{t=n} \right]^{-1}, /f$:
     *
     * where H is the local residual and c is the local state vector. The pseudo time is
     * denoted by t, where n denotes the current step index.
     *
     * \param [in] aControl 1D view of control variables, i.e. design variables
     * \param [in] aStateData state data manager
     * \param [in] aInvLocalJacobianT inverse of local Jacobian wrt local states
    *********************************************************************************
    void updateInverseLocalJacobian(const Plato::ScalarVector & aControl,
                                    const Plato::StateData& aStateData,
                                    Plato::ScalarArray3D& aInvLocalJacobianT)
    {
        auto tDhDc = mLocalResidualEq.gradient_c(aStateData.mCurrentGlobalState, aStateData.mPreviousGlobalState,
                                                 aStateData.mCurrentLocalState , aStateData.mPreviousLocalState,
                                                 aStateData.mCurrentProjPressGrad, aControl, aStateData.mTimeStep);
        auto tNumCells = mLocalResidualEq.numCells();
        Plato::inverse_matrix_workset<mNumLocalDofsPerCell, mNumLocalDofsPerCell>(tNumCells, tDhDc, aInvLocalJacobianT);
    }

    *****************************************************************************
     * \brief Update total gradient of performance criterion with respect to controls.
     * The total gradient is given by:
     *
     *  /f$ \left(\frac{df}{dz}\right)_{t=n} = \left(\frac{\partial{f}}{\partial{z}}\right)_{t=n}
     *         + \left(\frac{\partial{R}}{\partial{z}}\right)_{t=n}^{T}\lambda_{t=n}
     *         + \left(\frac{\partial{H}}{\partial{z}}\right)_{t=n}^{T}\gamma_{t=n}
     *         + \left(\frac{\partial{P}}{\partial{z}}\right)_{t=n}^{T}\mu_{t=n} /f$,
     *
     * where R is the global residual, H is the local residual, P is the projection residual,
     * and /f$\lambda/f$ is the global adjoint vector, /f$\gamma/f$ is the local adjoint vector,
     * and /f$\mu/f$ is the projection adjoint vector. The pseudo time is denoted by t, where n
     * denotes the current step index.
     *
     * \param [in] aControl 1D view of control variables, i.e. design variables
     * \param [in] aStateData state data manager
     * \param [in] aAdjointData adjoint data manager
     * \param [in/out] aGradient total derivative wrt controls
    *********************************************************************************
    void updateGradientControl(const Plato::ScalarVector &aControl,
                               const Plato::StateData &aStateData,
                               const Plato::AdjointData &aAdjointData,
                               Plato::ScalarVector &aGradient)
    {

        // add global adjoint contribution to gradient, i.e. DfDz += (DrDz)^T * lambda
        auto tDrDz_T = mGlobalResidualEq.gradient_z(aStateData.mCurrentGlobalState, aStateData.mPreviousGlobalState,
                                                    aStateData.mCurrentLocalState, aStateData.mPreviousLocalState,
                                                    aStateData.mCurrentProjPressGrad, aControl, aStateData.mTimeStep);
        Plato::MatrixTimesVectorPlusVector(tDrDz_T, aAdjointData.mCurrentGlobalAdjoint, aGradient);

        // add projection adjoint contribution to gradient, i.e. DfDz += (DpDz)^T * mu
        Plato::extract<mNumGlobalDofsPerNode, mPressureDofOffset>(aStateData.mCurrentGlobalState, mProjectedPressure);
        auto tDpDz_T = mProjectionEq.gradient_z(aStateData.mCurrentProjPressGrad, mProjectedPressure, aControl, aStateData.mTimeStep); //todo: ask josh about the order of the input arguments
        Plato::MatrixTimesVectorPlusVector(tDpDz_T, mProjectionAdjoint, aGradient);

        // add contribution from local residual to gradient, i.e. DfDz += (DhDz)^T * gamma
        auto tDhDz_T = mLocalResidualEq.gradient_z(aStateData.mCurrentGlobalState, aStateData.mPreviousGlobalState,
                                                 aStateData.mCurrentLocalState, aStateData.mPreviousLocalState,
                                                 aStateData.mCurrentProjPressGrad, aControl, aStateData.mTimeStep);
        auto tNumCells = mLocalResidualEq.numCells();
        Plato::Scalar tAlpha = 1.0; Plato::Scalar tBeta = 0.0;
        Plato::ScalarMultiVector tDhDzTimesLocalAdjoint("Local Gradient times adjoint", tNumCells, mNumNodesPerCell);
        Plato::matrix_times_vector_workset("T", tNumCells, tAlpha, tDhDz_T, aAdjointData.mCurrentLocalAdjoint, tBeta, tDhDzTimesLocalAdjoint);
        mWorksetBase.assembleScalarGradientZ(tDhDzTimesLocalAdjoint, aGradient);
    }

    *****************************************************************************
     * \brief Update total gradient of performance criterion with respect to configuration.
     * The total gradient is given by:
     *
     *  /f$ \left(\frac{df}{dz}\right)_{t=n} = \left(\frac{\partial{f}}{\partial{x}}\right)_{t=n}
     *         + \left(\frac{\partial{R}}{\partial{x}}\right)_{t=n}^{T}\lambda_{t=n}
     *         + \left(\frac{\partial{H}}{\partial{x}}\right)_{t=n}^{T}\gamma_{t=n}
     *         + \left(\frac{\partial{P}}{\partial{x}}\right)_{t=n}^{T}\mu_{t=n} /f$,
     *
     * where R is the global residual, H is the local residual, P is the projection residual,
     * and /f$\lambda/f$ is the global adjoint vector, /f$\gamma/f$ is the local adjoint vector,
     * x denotes the configuration variables, and /f$\mu/f$ is the projection adjoint vector.
     * The pseudo time is denoted by t, where n denotes the current step index.
     *
     * \param [in] aControl 1D view of control variables, i.e. design variables
     * \param [in] aStateData state data manager
     * \param [in] aAdjointData adjoint data manager
     * \param [in/out] aGradient total derivative wrt configuration
    *********************************************************************************
    void updateGradientConfiguration(const Plato::ScalarVector &aControl,
                                     const Plato::StateData &aStateData,
                                     const Plato::AdjointData &aAdjointData,
                                     Plato::ScalarVector &aGradient)
    {

        // add global adjoint contribution to gradient, i.e. DfDx += (DrDx)^T * lambda
        auto tDrDx_T = mGlobalResidualEq.gradient_x(aStateData.mCurrentGlobalState, aStateData.mPreviousGlobalState,
                                                    aStateData.mCurrentLocalState, aStateData.mPreviousLocalState,
                                                    aStateData.mCurrentProjPressGrad, aControl, aStateData.mTimeStep);
        Plato::MatrixTimesVectorPlusVector(tDrDx_T, aAdjointData.mCurrentGlobalAdjoint, aGradient);

        // add projection adjoint contribution to gradient, i.e. DfDx += (DpDx)^T * mu
        Plato::extract<mNumGlobalDofsPerNode, mPressureDofOffset>(aStateData.mCurrentGlobalState, mProjectedPressure);
        auto tDpDx_T = mProjectionEq.gradient_x(aStateData.mCurrentProjPressGrad,
                                                mProjectedPressure,
                                                aControl, aStateData.mTimeStep); //todo: ask josh about the order of the input arguments
        Plato::MatrixTimesVectorPlusVector(tDpDx_T, mProjectionAdjoint, aGradient);

        // add contribution from local residual to gradient, i.e. DfDx += (DhDx)^T * gamma
        auto tDhDx_T = mLocalResidualEq.gradient_x(aStateData.mCurrentGlobalState, aStateData.mPreviousGlobalState,
                                                   aStateData.mCurrentLocalState, aStateData.mPreviousLocalState,
                                                   aStateData.mCurrentProjPressGrad, aControl, aStateData.mTimeStep);
        auto tNumCells = mLocalResidualEq.numCells();
        Plato::Scalar tAlpha = 1.0; Plato::Scalar tBeta = 0.0;
        Plato::ScalarMultiVector tDhDxTimesLocalAdjoint("Local Gradient times adjoint", tNumCells, mNumNodesPerCell);
        Plato::matrix_times_vector_workset("T", tNumCells, tAlpha, tDhDx_T, aAdjointData.mCurrentLocalAdjoint, tBeta, tDhDxTimesLocalAdjoint);
        mWorksetBase.assembleVectorGradientX(tDhDxTimesLocalAdjoint, aGradient);
    }

    *****************************************************************************
     * \brief Update projection adjoint vector using the following equation:
     *
     *  /f$ \mu_{n} = -\left( \left(\frac{\partial{P}}{\partial{\pi}}\right)_{t=n}^{T} \right)^{-1}
     *                \left[ \left(\frac{\partial{R}}{\partial{\pi}}\right)_{t=n}^{T}\lambda_n \right] /f$,
     *
     * where R is the global residual, P is the projection residual, and /f$\pi/f$ is
     * the projection adjoint vector. The pseudo time is denoted by t, where n denotes
     * the current step index.
     *
     * \param [in] aControl 1D view of control variables, i.e. design variables
     * \param [in] aStateData state data manager
     * \param [in] aAdjointData adjoint data manager
    *********************************************************************************
    void updateProjectionAdjoint(const Plato::ScalarVector & aControl,
                                 const Plato::StateData& aStateData,
                                 Plato::AdjointData& aAdjointData)
    {
        auto tDrDp_T = mProjectionEq.gradient_n_T(aStateData.mCurrentProjPressGrad, mProjectedPressure, aControl, aStateData.mTimeStep);
        Plato::fill(static_cast<Plato::Scalar>(0.0), mProjResidual);
        Plato::MatrixTimesVectorPlusVector(tDrDp_T, aAdjointData.mCurrentGlobalAdjoint, mProjResidual);
        Plato::scale(static_cast<Plato::Scalar>(-1), mProjResidual);
        Plato::fill(static_cast<Plato::Scalar>(0.0), mProjectionAdjoint);
        Plato::Solve::RowSummed<SimplexPhysics::mNumSpatialDims>(mProjectionJacobian, mProjectionAdjoint, mProjResidual);
    }

    *****************************************************************************
     * \brief Update local adjoint vector using the following equation:
     *
     *  /f$ \gamma_n = -\left( \left(\frac{\partial{H}}{\partial{c}}\right)_{t=n}^{T} \right)^{-1}
     *                  \left[ \left(\frac{\partial{R}}{\partial{c}}\right)_{t=n}^{T}\lambda_n +
     *                  \left(\frac{\partial{f}}{\partial{c}} + \frac{\partial{H}}{\partial{v}}
     *                  \right)_{t=n+1}^{T} \gamma_{n+1} \right] /f$,
     *
     * where R is the global residual, H is the local residual, u is the global state,
     * c is the local state, f is the performance criterion (e.g. objective function),
     * and /f$\gamma/f$ is the local adjoint vector. The pseudo time is denoted by t,
     * where n denotes the current step index and n+1 is the previous time step index.
     *
     * \param [in] aControl 1D view of control variables, i.e. design variables
     * \param [in] aStateData state data manager
     * \param [in] aInvLocalJacobianT inverse of transpose local Jacobian wrt local states
     * \param [in] aAdjointData adjoint data manager
     * \param [in] aCriterion performance criterion interface
    *********************************************************************************
    void updateLocalAdjoint(const Plato::ScalarVector & aControl,
                            const Plato::StateData& aStateData,
                            const Plato::ScalarArray3D& aInvLocalJacobianT,
                            Plato::AdjointData aAdjointData,
                            Plato::ScalarFunctionLocalHistBase& aCriterion)
    {
        // Get current global adjoint workset
        auto tNumCells = mLocalResidualEq.numCells();
        Plato::ScalarMultiVector tCurrentGlobalAdjointWS("Current Global Adjoint Workset", tNumCells, mNumGlobalDofsPerCell);
        mWorksetBase.worksetLocalState(aAdjointData.mCurrentGlobalAdjoint, tCurrentGlobalAdjointWS);

        // Compute Right Hand Side (RHS) = [ (tDrDc^T * tCurrentGlobalAdjoint) + ( tDfDc + (tDhDcp^T * tPrevLocalAdjoint) ) ]
        auto tDrDc = mGlobalResidualEq.gradient_c(aStateData.mCurrentGlobalState, aStateData.mPreviousGlobalState,
                                                  aStateData.mCurrentLocalState , aStateData.mPreviousLocalState,
                                                  aStateData.mCurrentProjPressGrad, aControl, aStateData.mTimeStep);
        Plato::Scalar tBeta = 0.0;
        Plato::Scalar tAlpha = 1.0;
        Plato::ScalarMultiVector tRHS("Local Adjoint RHS", tNumCells, mNumLocalDofsPerCell);
        Plato::matrix_times_vector_workset("T", tNumCells, tAlpha, tDrDc, tCurrentGlobalAdjointWS, tBeta, tRHS);

        auto tFinalStepIndex = mNumPseudoTimeSteps - static_cast<Plato::OrdinalType>(1);
        if(aStateData.mCurrentStepIndex != tFinalStepIndex)
        {

            // Get previous local adjoint workset
            const Plato::OrdinalType tPreviousTimeIndex = 0;
            Plato::ScalarMultiVector tPreviousLocalAdjointWS("Local Previous Adjoint Workset", tNumCells, mNumLocalDofsPerCell);
            mWorksetBase.worksetLocalState(aAdjointData.mPreviousLocalAdjoint, tPreviousLocalAdjointWS);

            // Add tDfDc + (tDhDcp^T * tPrevLocalAdjoint) to tRHS
            tBeta = 1.0;
            auto tDfDc = aCriterion.gradient_c(aStateData.mCurrentGlobalState, aStateData.mCurrentLocalState, aControl, aStateData.mTimeStep);
            Plato::update_vector_workset(tNumCells, tAlpha, tDfDc, tBeta, tRHS);
            auto tDhDcp = mLocalResidualEq.gradient_cp(aStateData.mCurrentGlobalState, aStateData.mPreviousGlobalState,
                                                       aStateData.mCurrentLocalState , aStateData.mPreviousLocalState,
                                                       aStateData.mCurrentProjPressGrad, aControl, aStateData.mTimeStep);
            tBeta = 1.0;
            Plato::matrix_times_vector_workset("T", tNumCells, tAlpha, tDhDcp, tPreviousLocalAdjointWS, tBeta, tRHS);
        }

        // Compute -[ Inv(tDhDc^T) * tRHS ] and update current local adjoint variables
        Plato::ScalarMultiVector tCurrentLocalAdjointWS("Local Previous Adjoint Workset", tNumCells, mNumLocalDofsPerCell);
        tAlpha = -1.0; tBeta = 0.0;
        Plato::matrix_times_vector_workset("T", tNumCells, tAlpha, aInvLocalJacobianT, tRHS, tBeta, tCurrentLocalAdjointWS);
        Plato::flatten_vector_workset<mNumLocalDofsPerCell>(tNumCells, tCurrentLocalAdjointWS, aAdjointData.mCurrentLocalAdjoint);
    }

    *****************************************************************************
     * \brief Update global adjoint vector
     *
     * \param [in] aControl 1D view of control variables, i.e. design variables
     * \param [in] aStateData state data manager
     * \param [in] aInvLocalJacobianT inverse of transpose local Jacobian wrt local states
     * \param [in] aAdjointData adjoint data manager
     * \param [in] aCriterion performance criterion interface
    *********************************************************************************
    void updateGlobalAdjoint(const Plato::ScalarVector & aControl,
                             const Plato::StateData& aStateData,
                             const Plato::ScalarArray3D& aInvLocalJacobianT,
                             Plato::AdjointData aAdjointData,
                             Plato::ScalarFunctionLocalHistBase& aCriterion)
    {
        // Assemble adjoint Jacobian into mGlobalJacobian
        this->assembleGlobalAdjointJacobian(aControl, aStateData, aInvLocalJacobianT);
        // Assemble right hand side vector into mGlobalResidual
        this->assembleGlobalAdjointForceVector(aControl, aStateData, aInvLocalJacobianT, aAdjointData, aCriterion);
        // Apply Dirichlet conditions
        this->applyConstraints(mGlobalJacobian, mGlobalResidual);
        // solve global system of equations
        Plato::fill(static_cast<Plato::Scalar>(0.0), aAdjointData.mCurrentGlobalAdjoint);
        Plato::Solve::Consistent<mNumGlobalDofsPerNode>(mGlobalJacobian, aAdjointData.mCurrentGlobalAdjoint, mGlobalResidual);
    }

    *****************************************************************************
     * \brief Assemble global adjoint right hand side vector, which is given by:
     *
     * /f$ \bm{f} = \frac{\partial{f}}{\partial{u}}\right)_{t=n} - \left(\frac{\partial{H}}{\partial{u}}
     * \right)_{t=n}^{T} * \left[ \left( \left(\frac{\partial{H}}{\partial{c}}\right)_{t=n}^{T} \right)^{-1} *
     * \left(\frac{\partial{f}}{\partial{c}} + \frac{\partial{H}}{\partial{v}}\right)_{t=n+1}^{T} \gamma_{n+1}
     * \right]/f$,
     *
     * where R is the global residual, H is the local residual, u is the global state,
     * c is the local state, f is the performance criterion (e.g. objective function),
     * and /f$\gamma/f$ is the local adjoint vector. The pseudo time is denoted by t,
     * where n denotes the current step index and n+1 is the previous time step index.
     *
     * \param [in] aControl 1D view of control variables, i.e. design variables
     * \param [in] aStateData state data manager
     * \param [in] aInvLocalJacobianT inverse of transpose local Jacobian wrt local states
     * \param [in] aAdjointData adjoint data manager
     * \param [in] aCriterion performance criterion interface
    *********************************************************************************
    void assembleGlobalAdjointForceVector(const Plato::ScalarVector &aControl,
                                          const Plato::StateData &aStateData,
                                          const Plato::ScalarArray3D& aInvLocalJacobianT,
                                          const Plato::AdjointData aAdjointData,
                                          Plato::ScalarFunctionLocalHistBase& aCriterion)
    {
        // TODO: MODIFY OUTPUT FROM LOCAL VECTOR FUNCTION, I WANT TO RETURN SCALAR_ARRAY_3D NOT THE AD TYPE MULTIVECTOR

        // Compute partial derivative of objective with respect to current global states
        auto tDfDu = aCriterion.gradient_u(aStateData.mCurrentGlobalState,
                                           aStateData.mCurrentLocalState,
                                           aControl, aStateData.mTimeStep);

        auto tFinalStepIndex = mNumPseudoTimeSteps - static_cast<Plato::OrdinalType>(1);
        if(aStateData.mCurrentStepIndex != tFinalStepIndex)
        {
            // Compute partial derivative of objective with respect to current local states
            auto tDfDc = aCriterion.gradient_c(aStateData.mCurrentGlobalState,
                                               aStateData.mCurrentLocalState,
                                               aControl, aStateData.mTimeStep);

            // Compute previous local adjoint workset
            auto tNumCells = mLocalResidualEq.numCells();
            Plato::ScalarMultiVector tLocalPrevAdjointWS("Local Previous Adjoint Workset", tNumCells, mNumLocalDofsPerCell);
            mWorksetBase.worksetLocalState(aAdjointData.mPreviousLocalAdjoint, tLocalPrevAdjointWS);

            // Compute tDfDc + (tDhDcp^T * tPrevLocalAdjoint)
            Plato::Scalar tBeta = 0.0;
            Plato::Scalar tAlpha = 1.0;
            Plato::ScalarMultiVector tWorkMultiVectorOneWS("Local State Work Workset", tNumCells, mNumLocalDofsPerCell);
            auto tDhDcp = mLocalResidualEq.gradient_cp(aStateData.mCurrentGlobalState, aStateData.mPreviousGlobalState,
                                                       aStateData.mCurrentLocalState , aStateData.mPreviousLocalState,
                                                       aStateData.mCurrentProjPressGrad, aControl, aStateData.mTimeStep);
            Plato::matrix_times_vector_workset("T", tNumCells, tAlpha, tDhDcp, tLocalPrevAdjointWS, tBeta, tWorkMultiVectorOneWS);
            tBeta = 1.0;
            Plato::update_vector_workset(tNumCells, tAlpha, tDfDc, tBeta, tWorkMultiVectorOneWS);

            // Compute Inv(tDhDc^T) * (tDfDc + tDhDcp^T * tPrevLocalAdjoint)
            Plato::ScalarMultiVector tWorkMultiVectorTwoWS("Local State Work Workset", tNumCells, mNumLocalDofsPerCell);
            Plato::matrix_times_vector_workset("T", tNumCells, tAlpha, aInvLocalJacobianT,
                                               tWorkMultiVectorOneWS, tBeta, tWorkMultiVectorTwoWS);

            // Compute tDhDu^T * [ Inv(tDhDc^T) * (tDfDc + tDhDcp^T * tPrevLocalAdjoint) ]
            auto tDhDu = mLocalResidualEq.gradient_u(aStateData.mCurrentGlobalState, aStateData.mPreviousGlobalState,
                                                     aStateData.mCurrentLocalState , aStateData.mPreviousLocalState,
                                                     aStateData.mCurrentProjPressGrad, aControl, aStateData.mTimeStep);
            Plato::ScalarMultiVector tRHS("Global State Right-Hand-Side Workset", tNumCells, mNumGlobalDofsPerCell);
            Plato::matrix_times_vector_workset("T", tNumCells, tAlpha, tDhDu, tWorkMultiVectorTwoWS, tBeta, tRHS);

            // Compute tDfDu - { tDhDu^T * [ INV(tDhDc^T) * (tDfDc + tDhDcp^T * tPrevLocalAdjoint) ] }
            tAlpha = -1.0; tBeta = 1.0;
            Plato::update_vector_workset(tNumCells, tAlpha, tRHS, tBeta, tDfDu);
        }

        Plato::fill(static_cast<Plato::Scalar>(0), mGlobalResidual);
        mWorksetBase.assembleVectorGradientU(tDfDu, mGlobalResidual);
        Plato::scale(static_cast<Plato::Scalar>(-1), mGlobalResidual)
    }

    *****************************************************************************
     * \brief Assemble global adjoint Jacobian, which is given by:
     *
     * /f$ A = \left(\frac{\partial{R}}{\partial{u}}\right)_{t=n}^{T} - \left(\frac{\partial{H}}{\partial{u}}
     * \right)_{t=n}^{T} * \left[ \left( \left(\frac{\partial{H}}{\partial{c}}\right)_{t=n}^{T} \right)^{-1} *
     * \left(\frac{\partial{R}}{\partial{v}}\right)_{t=n}^{T} \right] - \left(\frac{\partial{P}}{\partial{u}}
     * \right)_{t=n}^{T} * \left[ \left( \left(\frac{\partial{P}}{\partial{\pi}}\right)_{t=n}^{T} \right)^{-1}
     * \left(\frac{\partial{R}}{\partial{\pi}}\right)_{t=n}^{T} \right] /f$,
     *
     * where R is the global residual, H is the local residual, P is the projection residual,
     * u is the global state, and c is the local state, and /f$\pi/f$ is the projected pressure
     * gradient. The pseudo time is denoted by t, where n denotes the current step index.
     *
     * \param [in] aControl 1D view of control variables, i.e. design variables
     * \param [in] aStateData state data manager
     * \param [in] aInvLocalJacobianT inverse of transpose local Jacobian wrt local states
    *********************************************************************************
    void assembleGlobalAdjointJacobian(const Plato::ScalarVector &aControl,
                                       const Plato::StateData &aStateData,
                                       const Plato::ScalarArray3D& aInvLocalJacobianT)
    {
        // Compute tangent stiffness matrix, DRdu - dHdu * [ inverse(dHdc) * dRdc ], where R is the
        // global residual, H is the local residual, u are the global states, and c are the local states.
        // Note: tangent stiffness matrix is symmetric; hence, the tangent stiffness matrix assembly
        // routine for forward and adjoint problems are the same
        this->assembleTangentStiffnessMatrix(aControl, aStateData, aInvLocalJacobianT);

        // Compute assembled Jacobian wrt projected pressure gradient
        Plato::extract<mNumGlobalDofsPerNode, mPressureDofOffset>(aStateData.mCurrentGlobalState, mProjectedPressure);
        mProjectionJacobian = mProjectionEq.gradient_u(aStateData.mCurrentProjPressGrad,
                                                       mProjectedPressure, aControl, aStateData.mTimeStep);

        // Compute dPdn^T: Assembled transpose of partial of projection residual wrt projected pressure gradient
        auto tDpDn_T = mProjectionEq.gradient_n_T(aStateData.mCurrentProjPressGrad,
                                                  mProjectedPressure, aControl, aStateData.mTimeStep);

        // compute dgdPI^T: Assembled transpose of partial of global residual wrt projected pressure gradient
        auto tDrDn_T = mGlobalResidualEq.gradient_n_T(aStateData.mCurrentGlobalState, aStateData.mPreviousGlobalState,
                                                      aStateData.mCurrentLocalState, aStateData.mPreviousLocalState,
                                                      aStateData.mCurrentProjPressGrad, aControl, aStateData.mTimeStep);

        // Compute Jacobian for adjoint problem: K_t - dPdn_T * (dPdn)^-1 * dRdn_T, where K_t
        // is the tangent stiffness matrix, P is the projection problem residual, R is the
        // global residual, and n is the pressure gradient
        Plato::Condense(mGlobalJacobian  K_t , tDpDn_T, mProjectionJacobian,  tDrDn_T, mPressureDofOffset  row offset );
    }

    *****************************************************************************
     * \brief Compute Schur complement, which is defined as /f$ A = \frac{\partial{R}}
     * {\partial{c}} * \left[ \left(\frac{\partial{H}}{\partial{c}}\right)^{-1} *
     * \frac{\partial{H}}{\partial{u}} \right] /f$, where R is the global residual, H
     * is the local residual, u are the global states, and c are the local states.
     *
     * \param [in] aControl 1D view of control variables, i.e. design variables
     * \param [in] aStateData state data manager
     * \param [in] aInvLocalJacobianT inverse of transpose local Jacobian wrt local states
     * \return 3D view with Schur complement per cell
    *********************************************************************************
    Plato::ScalarArray3D computeSchurComplement(const Plato::ScalarVector & aControl,
                                                const Plato::StateData& aStateData,
                                                const Plato::ScalarArray3D& aInvLocalJacobianT)
    {
        // TODO: MODIFY OUTPUT FROM LOCAL VECTOR FUNCTION, I WANT TO RETURN SCALAR_ARRAY_3D NOT THE AD TYPE MULTIVECTOR

        // Compute cell Jacobian of the local residual with respect to the current global state WorkSet (WS)
        auto tDhDu = mLocalResidualEq.gradient_u(aStateData.mCurrentGlobalState, aStateData.mPreviousGlobalState,
                                                 aStateData.mCurrentLocalState, aStateData.mPreviousLocalState,
                                                 aStateData.mCurrentProjPressGrad, aControl, aStateData.mTimeStep);

        // Compute cell C = (dH/dc)^{-1}*dH/du, where H is the local residual, c are the local states and u are the global states
        const Plato::Scalar tBeta = 1.0;
        const Plato::Scalar tAlpha = 1.0;
        auto tNumCells = mLocalResidualEq.numCells();
        Plato::ScalarArray3D tInvDhDcTimesDhDu("InvDhDu times DhDu", tNumCells, mNumLocalDofsPerCell, mNumLocalDofsPerCell);
        Plato::multiply_matrix_workset("N", "N", tNumCells, tAlpha, aInvLocalJacobianT, tDhDu, tBeta, tInvDhDcTimesDhDu);


        // Compute cell Jacobian of the global residual with respect to the current local state WorkSet (WS)
        auto tDrDc = mGlobalResidualEq.gradient_c(aStateData.mCurrentGlobalState, aStateData.mPreviousGlobalState,
                                                  aStateData.mCurrentLocalState, aStateData.mPreviousLocalState,
                                                  aStateData.mCurrentProjPressGrad, aControl, aStateData.mTimeStep);

        // Compute cell Schur = dR/dc * (dH/dc)^{-1} * dH/du, where H is the local residual,
        // R is the global residual, c are the local states and u are the global states
        Plato::ScalarArray3D tOutput("Schur Complement", tNumCells, mNumLocalDofsPerCell, mNumLocalDofsPerCell);
        Plato::multiply_matrix_workset("N", "N", tNumCells, tAlpha, tDrDc, tInvDhDcTimesDhDu, tBeta, tOutput);
        return tOutput;
    }

    *****************************************************************************
     * \brief Assemble tangent stiffness matrix, which is defined as /f$ K_{T} =
     * \frac{\partial{R}}{\partial{u}} - \frac{\partial{R}}{\partial{c}} * \left[
     * \left(\frac{\partial{H}}{\partial{c}}\right)^{-1} * \frac{\partial{H}}{\partial{u}}
     * \right] /f$, where R is the global residual, H is the local residual, u are
     * the global states, and c are the local states.
     *
     * \param [in] aControl 1D view of control variables, i.e. design variables
     * \param [in] aStateData state data manager
     * \param [in] aInvLocalJacobianT inverse of transpose local Jacobian wrt local states
    *********************************************************************************
    void assembleTangentStiffnessMatrix(const Plato::ScalarVector & aControl,
                                        const Plato::StateData& aStateData,
                                        const Plato::ScalarArray3D& aInvLocalJacobianT)
    {
        // Compute cell Schur Complement, i.e. dR/dc * (dH/dc)^{-1} * dH/du, where H is the local
        // residual, R is the global residual, c are the local states and u are the global states
        auto tSchurComplement = this->computeSchurComplement(aControl, aStateData, aInvLocalJacobianT);

        // Compute cell Jacobian of the global residual with respect to the current global state WorkSet (WS)
        auto tDrDu = mGlobalResidualEq.gradient_u(aStateData.mCurrentGlobalState, aStateData.mPreviousGlobalState,
                                                  aStateData.mCurrentLocalState, aStateData.mPreviousLocalState,
                                                  aStateData.mCurrentProjPressGrad, aControl, aStateData.mTimeStep);

        // Add cell Schur complement to dR/du, where R is the global residual and u are the global states
        const Plato::Scalar tBeta = 1.0;
        const Plato::Scalar tAlpha = -1.0;
        auto tNumCells = mGlobalResidualEq.numCells();
        Plato::update_matrix_workset(tNumCells, tAlpha, tSchurComplement, tBeta, tDrDu);

        auto tJacobianEntries = mGlobalJacobian->entries();
        Plato::assemble_jacobian(tNumCells, mNumGlobalDofsPerCell, mNumGlobalDofsPerCell,
                                 *mGlobalJacEntryOrdinal, tDrDu, tJacobianEntries);
    }

    *****************************************************************************
     * \brief Set Dirichlet degrees of freedom in global state increment view to zero
     * \param [in] aGlobalStateIncrement 1D view of global state increments, i.e. Newton-Raphson solution
    *********************************************************************************
    void zeroDirichletDofs(Plato::ScalarVector & aGlobalStateIncrement)
    {
        auto tDirichletDofs = mDirichletDofs;
        auto tNumDirichletDofs = mDirichletDofs.size();
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumDirichletDofs), LAMBDA_EXPRESSION(const Plato::OrdinalType & aDofOrdinal)
        {
            auto tLocalDofIndex = tDirichletDofs[aDofOrdinal];
            aGlobalStateIncrement(tLocalDofIndex) = 0.0;
        },"zero global state increment dirichlet dofs");
    }

    *****************************************************************************
     * \brief Check Newton-Raphson solver convergence criterion
     * \param [in] aNewtonIteration current Newton-Raphson iteration
     * \return boolean flag, indicates if Newton-Raphson solver converged
    *********************************************************************************
    bool checkNewtonRaphsonStoppingCriteria(const Plato::OrdinalType & aNewtonIteration)
    {
        bool tStop = false;
        auto tNormResidual = Plato::norm(mGlobalResidual);
        if(aNewtonIteration == static_cast<Plato::OrdinalType>(0))
        {
            mInitialNormResidual = tNormResidual;
        }
        else
        {
            auto tStoppingMeasure = tNormResidual / mInitialNormResidual; // compute relative stopping criterion
            if(tStoppingMeasure < mNewtonRaphsonStopTolerance)
            {
                tStop = true;
            }
        }
        return (tStop);
    }

// private functions
private:
    *****************************************************************************
     * \brief Initialize member data
     * \param [in] aMesh mesh database
     * \param [in] aMeshSets side sets database
     * \param [in] aInputParams input parameters database
     *********************************************************************************
    void initialize(Omega_h::Mesh& aMesh, Omega_h::MeshSets& aMeshSets, Teuchos::ParameterList& aInputParams)
    {
        this->allocateObjectiveFunction(aMesh, aMeshSets, aInputParams);
        this->allocateConstraintFunction(aMesh, aMeshSets, aInputParams);

        // Allocate Jacobian Matrix
        mGlobalJacobian = Plato::CreateBlockMatrix<Plato::CrsMatrixType, mSpatialDim, mNumGlobalDofsPerNode>(&aMesh);
        mGlobalJacEntryOrdinal = std::make_shared<Plato::BlockMatrixEntryOrdinal<mSpatialDim, mNumGlobalDofsPerNode>>(mGlobalJacobian, &aMesh);

        // Parse Dirichlet boundary conditions
        Plato::EssentialBCs<SimplexPhysics> tDirichletBCs(aInputParams.sublist("Essential Boundary Conditions", false));
        tDirichletBCs.get(aMeshSets, mDirichletDofs, mDirichletValues);
    }

    *****************************************************************************
     * \brief Allocate objective function interface and adjoint containers
     * \param [in] aMesh mesh database
     * \param [in] aMeshSets side sets database
     * \param [in] aInputParams input parameters database
     *********************************************************************************
    void allocateObjectiveFunction(Omega_h::Mesh& aMesh, Omega_h::MeshSets& aMeshSets, Teuchos::ParameterList& aInputParams)
    {
        if(aInputParams.isType<std::string>("Objective"))
        {
            auto tObjectiveType = aInputParams.get<std::string>("Objective");
            Plato::ScalarFunctionIncBaseFactory<SimplexPhysics> tObjectiveFunctionFactory;
            mObjective = tObjectiveFunctionFactory.create(aMesh, aMeshSets, mDataMap, aInputParams, tObjectiveType);

            // Allocate adjoint variable containers
            mProjectionAdjoint("Projected Residual Adjoint", mProjectionEq.size());
            mLocalAdjoint("Local Adjoint", static_cast<Plato::OrdinalType>(2), mLocalResidualEq.size());
            mGlobalAdjoint("Global Adjoint", static_cast<Plato::OrdinalType>(2), mGlobalResidualEq.size());
        }
    }

    *****************************************************************************
     * \brief Allocate constraint function interface and adjoint containers
     * \param [in] aMesh mesh database
     * \param [in] aMeshSets side sets database
     * \param [in] aInputParams input parameters database
     *********************************************************************************
    void allocateConstraintFunction(Omega_h::Mesh& aMesh, Omega_h::MeshSets& aMeshSets, Teuchos::ParameterList& aInputParams)
    {
        if(aInputParams.isType<std::string>("Constraint"))
        {
            Plato::ScalarFunctionBaseFactory<SimplexPhysics> tContraintFunctionFactory;
            auto tConstraintType = aInputParams.get<std::string>("Constraint");
            mConstraint = tContraintFunctionFactory.create(aMesh, aMeshSets, mDataMap, aInputParams, tConstraintType);
        }
    }
};
// class PlasticityProblem

*/


}
// namespace Plato

namespace ElastoPlasticityTest
{

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, ElastoPlasticity_FlattenVectorWorkset_Errors)
{
    // CALL FUNCTION - TEST tLocalStateWorset IS EMPTY
    Plato::ScalarVector tAssembledLocalState;
    Plato::ScalarMultiVector tLocalStateWorset;
    constexpr Plato::OrdinalType tNumCells = 1;
    constexpr Plato::OrdinalType tNumLocalDofsPerCell = 14;
    TEST_THROW(Plato::flatten_vector_workset<tNumLocalDofsPerCell>(tNumCells, tLocalStateWorset, tAssembledLocalState), std::runtime_error);

    // CALL FUNCTION - TEST tAssembledLocalState IS EMPTY
    tLocalStateWorset = Plato::ScalarMultiVector("local state WS", tNumCells, tNumLocalDofsPerCell);
    TEST_THROW(Plato::flatten_vector_workset<tNumLocalDofsPerCell>(tNumCells, tLocalStateWorset, tAssembledLocalState), std::runtime_error);

    // CALL FUNCTION - TEST NUMBER OF CELLS IS EMPTY
    constexpr Plato::OrdinalType tEmptyNumCells = 0;
    tAssembledLocalState = Plato::ScalarVector("assembled local state", tNumCells * tNumLocalDofsPerCell);
    TEST_THROW(Plato::flatten_vector_workset<tNumLocalDofsPerCell>(tEmptyNumCells, tLocalStateWorset, tAssembledLocalState), std::runtime_error);
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, ElastoPlasticity_FlattenVectorWorkset)
{
    // PREPARE DATA
    constexpr Plato::OrdinalType tNumCells = 3;
    constexpr Plato::OrdinalType tNumLocalDofsPerCell = 14;
    Plato::ScalarMultiVector tLocalStateWorset("local state WS", tNumCells, tNumLocalDofsPerCell);
    auto tHostLocalStateWorset = Kokkos::create_mirror(tLocalStateWorset);

    for (size_t tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
        for (size_t tDofIndex = 0; tDofIndex < tNumLocalDofsPerCell; tDofIndex++)
        {
            tHostLocalStateWorset(tCellIndex, tDofIndex) = (tNumLocalDofsPerCell * tCellIndex) + (tDofIndex + 1.0);
            //printf("(%d,%d) = %f\n", tCellIndex, tDofIndex, tHostLocalStateWorset(tCellIndex, tDofIndex));
        }
    }
    Kokkos::deep_copy(tLocalStateWorset, tHostLocalStateWorset);

    Plato::ScalarVector tAssembledLocalState("assembled local state", tNumCells * tNumLocalDofsPerCell);

    // CALL FUNCTION
    TEST_NOTHROW(Plato::flatten_vector_workset<tNumLocalDofsPerCell>(tNumCells, tLocalStateWorset, tAssembledLocalState));

    // TEST RESULTS
    constexpr Plato::Scalar tTolerance = 1e-4;
    auto tHostAssembledLocalState = Kokkos::create_mirror(tAssembledLocalState);
    Kokkos::deep_copy(tHostAssembledLocalState, tAssembledLocalState);
    std::vector<std::vector<Plato::Scalar>> tGold =
      {{1,2,3,4,5,6,7,8,9,10,11,12,13,14},
       {15,16,17,18,19,20,21,22,23,24,25,26,27,28},
       {29,30,31,32,33,34,35,36,37,38,39,40,41,42}};
    for(Plato::OrdinalType tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
        const auto tDofOffset = tCellIndex * tNumLocalDofsPerCell;
        for(Plato::OrdinalType tDofIndex = 0; tDofIndex < tNumLocalDofsPerCell; tDofIndex++)
        {
            //printf("(%d,%d) = %f\n", tCellIndex, tDofIndex, tHostAssembledLocalState(tDofOffset + tDofIndex));
            TEST_FLOATING_EQUALITY(tHostAssembledLocalState(tDofOffset + tDofIndex), tGold[tCellIndex][tDofIndex], tTolerance);
        }
    }
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, ElastoPlasticity_fill3DView_Error)
{
    // PREPARE DATA
    constexpr Plato::OrdinalType tNumRows = 14;
    constexpr Plato::OrdinalType tNumCols = 14;
    constexpr Plato::OrdinalType tNumCells = 2;

    // CALL FUNCTION - TEST tMatrixWorkSet IS EMPTY
    constexpr Plato::Scalar tAlpha = 2.0;
    Plato::ScalarArray3D tMatrixWorkSet;
    TEST_THROW( (Plato::fill_3D_workset<tNumRows, tNumCols>(tNumCells, tAlpha, tMatrixWorkSet)), std::runtime_error );

    // CALL FUNCTION - TEST tNumCells IS ZERO
    Plato::OrdinalType tBadNumCells = 0;
    tMatrixWorkSet = Plato::ScalarArray3D("Matrix A WS", tNumCells, tNumRows, tNumCols);
    TEST_THROW( (Plato::fill_3D_workset<tNumRows, tNumCols>(tBadNumCells, tAlpha, tMatrixWorkSet)), std::runtime_error );

    // CALL FUNCTION - TEST tNumCells IS NEGATIVE
    tBadNumCells = -1;
    TEST_THROW( (Plato::fill_3D_workset<tNumRows, tNumCols>(tBadNumCells, tAlpha, tMatrixWorkSet)), std::runtime_error );
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, ElastoPlasticity_fill3DView)
{
    // PREPARE DATA
    constexpr Plato::OrdinalType tNumRows = 14;
    constexpr Plato::OrdinalType tNumCols = 14;
    constexpr Plato::OrdinalType tNumCells = 2;
    Plato::ScalarArray3D tA("Matrix A WS", tNumCells, tNumRows, tNumCols);

    // CALL FUNCTION
    Plato::Scalar tAlpha = 2.0;
    TEST_NOTHROW( (Plato::fill_3D_workset<tNumRows, tNumCols>(tNumCells, tAlpha, tA)) );

    // TEST RESULTS
    constexpr Plato::Scalar tGold = 2.0;
    constexpr Plato::Scalar tTolerance = 1e-4;
    auto tHostA = Kokkos::create_mirror(tA);
    Kokkos::deep_copy(tHostA, tA);
    for(Plato::OrdinalType tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
        for(Plato::OrdinalType tRowIndex = 0; tRowIndex < tNumRows; tRowIndex++)
        {
            for(Plato::OrdinalType tColIndex = 0; tColIndex < tNumCols; tColIndex++)
            {
                //printf("(%d,%d,%d) = %f\n", aCellOrdinal, tRowIndex, tColIndex, tHostA(tCellIndex, tRowIndex, tColIndex));
                TEST_FLOATING_EQUALITY(tHostA(tCellIndex, tRowIndex, tColIndex), tGold, tTolerance);
            }
        }
    }
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, ElastoPlasticity_UpdateMatrixWorkset_Error)
{
    // CALL FUNCTION - INPUT VIEW IS EMPTY
    Plato::ScalarArray3D tB;
    Plato::ScalarArray3D tA;
    constexpr Plato::OrdinalType tNumCells = 2;
    Plato::Scalar tAlpha = 1; Plato::Scalar tBeta = 3;
    TEST_THROW( (Plato::update_matrix_workset(tNumCells, tAlpha, tA, tBeta, tB)), std::runtime_error );

    // CALL FUNCTION - OUTPUT VIEW IS EMPTY
    Plato::OrdinalType tNumRows = 4;
    Plato::OrdinalType tNumCols = 4;
    tA = Plato::ScalarArray3D("Matrix A WS", tNumCells, tNumRows, tNumCols);
    TEST_THROW( (Plato::update_matrix_workset(tNumCells, tAlpha, tA, tBeta, tB)), std::runtime_error );

    // CALL FUNCTION - ROW DIM MISTMATCH
    tNumRows = 3;
    Plato::ScalarArray3D tC = Plato::ScalarArray3D("Matrix C WS", tNumCells, tNumRows, tNumCols);
    tNumRows = 4;
    Plato::ScalarArray3D tD = Plato::ScalarArray3D("Matrix D WS", tNumCells, tNumRows, tNumCols);
    TEST_THROW( (Plato::update_matrix_workset(tNumCells, tAlpha, tC, tBeta, tD)), std::runtime_error );

    // CALL FUNCTION - COLUMN DIM MISTMATCH
    tNumCols = 5;
    Plato::ScalarArray3D tE = Plato::ScalarArray3D("Matrix E WS", tNumCells, tNumRows, tNumCols);
    TEST_THROW( (Plato::update_matrix_workset(tNumCells, tAlpha, tD, tBeta, tE)), std::runtime_error );

    // CALL FUNCTION - NEGATIVE NUMBER OF CELLS
    tNumRows = 4; tNumCols = 4;
    Plato::OrdinalType tBadNumCells = -1;
    tB = Plato::ScalarArray3D("Matrix B WS", tNumCells, tNumRows, tNumCols);
    TEST_THROW( (Plato::update_matrix_workset(tBadNumCells, tAlpha, tA, tBeta, tB)), std::runtime_error );

    // CALL FUNCTION - ZERO NUMBER OF CELLS
    tBadNumCells = 0;
    TEST_THROW( (Plato::update_matrix_workset(tBadNumCells, tAlpha, tA, tBeta, tB)), std::runtime_error );
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, ElastoPlasticity_UpdateMatrixWorkset)
{
    // PREPARE DATA
    constexpr Plato::OrdinalType tNumRows = 14;
    constexpr Plato::OrdinalType tNumCols = 14;
    constexpr Plato::OrdinalType tNumCells = 2;
    Plato::ScalarArray3D tA("Matrix A WS", tNumCells, tNumRows, tNumCols);
    Plato::Scalar tAlpha = 2;
    TEST_NOTHROW( (Plato::fill_3D_workset<tNumRows, tNumCols>(tNumCells, tAlpha, tA)) );

    tAlpha = 1;
    Plato::ScalarArray3D tB("Matrix A WS", tNumCells, tNumRows, tNumCols);
    TEST_NOTHROW( (Plato::fill_3D_workset<tNumRows, tNumCols>(tNumCells, tAlpha, tB)) );

    // CALL FUNCTION
    tAlpha = 2;
    Plato::Scalar tBeta = 3;
    TEST_NOTHROW( (Plato::update_matrix_workset(tNumCells, tAlpha, tA, tBeta, tB)) );

    // TEST RESULTS
    constexpr Plato::Scalar tGold = 7.0;
    constexpr Plato::Scalar tTolerance = 1e-4;
    auto tHostB = Kokkos::create_mirror(tB);
    Kokkos::deep_copy(tHostB, tB);
    for(Plato::OrdinalType tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
        for(Plato::OrdinalType tRowIndex = 0; tRowIndex < tNumRows; tRowIndex++)
        {
            for(Plato::OrdinalType tColIndex = 0; tColIndex < tNumCols; tColIndex++)
            {
                //printf("(%d,%d,%d) = %f\n", aCellOrdinal, tRowIndex, tColIndex, tHostB(tCellIndex, tRowIndex, tColIndex));
                TEST_FLOATING_EQUALITY(tHostB(tCellIndex, tRowIndex, tColIndex), tGold, tTolerance);
            }
        }
    }
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, ElastoPlasticity_UpdateVectorWorkset_Error)
{
    // CALL FUNCTION - DIMENSION MISMATCH
    Plato::OrdinalType tNumDofsPerCell = 3;
    constexpr Plato::OrdinalType tNumCells = 2;
    Plato::ScalarMultiVector tVecX("vector X WS", tNumCells, tNumDofsPerCell);
    tNumDofsPerCell = 4;
    Plato::ScalarMultiVector tVecY("vector Y WS", tNumCells, tNumDofsPerCell);
    Plato::Scalar tAlpha = 1; Plato::Scalar tBeta = 3;
    TEST_THROW( (Plato::update_vector_workset(tNumCells, tAlpha, tVecX, tBeta, tVecY)), std::runtime_error );


    // CALL FUNCTION - NEGATIVE NUMBER OF CELLS
    Plato::OrdinalType tBadNumCells = -1;
    Plato::ScalarMultiVector tVecZ("vector Y WS", tNumCells, tNumDofsPerCell);
    TEST_THROW( (Plato::update_vector_workset(tBadNumCells, tAlpha, tVecY, tBeta, tVecZ)), std::runtime_error );

    // CALL FUNCTION - ZERO NUMBER OF CELLS
    tBadNumCells = 0;
    TEST_THROW( (Plato::update_vector_workset(tBadNumCells, tAlpha, tVecY, tBeta, tVecZ)), std::runtime_error );
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, ElastoPlasticity_UpdateVectorWorkset)
{
    // PREPARE DATA
    constexpr Plato::OrdinalType tNumCells = 2;
    constexpr Plato::OrdinalType tNumLocalDofsPerCell = 6;
    Plato::ScalarMultiVector tVecX("vector X WS", tNumCells, tNumLocalDofsPerCell);
    Plato::ScalarMultiVector tVecY("vector Y WS", tNumCells, tNumLocalDofsPerCell);
    auto tHostVecX = Kokkos::create_mirror(tVecX);
    auto tHostVecY = Kokkos::create_mirror(tVecY);

    for (size_t tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
        for (size_t tDofIndex = 0; tDofIndex < tNumLocalDofsPerCell; tDofIndex++)
        {
            tHostVecX(tCellIndex, tDofIndex) = (tNumLocalDofsPerCell * tCellIndex) + (tDofIndex + 1.0);
            tHostVecY(tCellIndex, tDofIndex) = (tNumLocalDofsPerCell * tCellIndex) + (tDofIndex + 1.0);
            //printf("X(%d,%d) = %f\n", tCellIndex, tDofIndex, tHostVecX(tCellIndex, tDofIndex));
            //printf("Y(%d,%d) = %f\n", tCellIndex, tDofIndex, tHostVecY(tCellIndex, tDofIndex));
        }
    }
    Kokkos::deep_copy(tVecX, tHostVecX);
    Kokkos::deep_copy(tVecY, tHostVecY);

    // CALL FUNCTION
    Plato::Scalar tAlpha = 1; Plato::Scalar tBeta = 2;
    TEST_NOTHROW( (Plato::update_vector_workset(tNumCells, tAlpha, tVecX, tBeta, tVecY)) );

    // TEST OUTPUT
    constexpr Plato::Scalar tTolerance = 1e-4;
    tHostVecY = Kokkos::create_mirror(tVecY);
    Kokkos::deep_copy(tHostVecY, tVecY);
    std::vector<std::vector<Plato::Scalar>> tGold =
      {{3, 6, 9, 12, 15, 18}, {21, 24, 27, 30, 33, 36}};
    for(Plato::OrdinalType tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
        for(Plato::OrdinalType tDofIndex = 0; tDofIndex < tNumLocalDofsPerCell; tDofIndex++)
        {
            //printf("(%d,%d) = %f\n", tCellIndex, tDofIndex, tHostVecY(tCellIndex, tDofIndex));
            TEST_FLOATING_EQUALITY(tHostVecY(tCellIndex, tDofIndex), tGold[tCellIndex][tDofIndex], tTolerance);
        }
    }
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, ElastoPlasticity_MultiplyMatrixWorkset_Error)
{
    // PREPARE DATA
    Plato::ScalarArray3D tA;
    Plato::ScalarArray3D tB;
    Plato::ScalarArray3D tC;

    // CALL FUNCTION - A IS EMPTY
    constexpr Plato::OrdinalType tNumCells = 2;
    Plato::Scalar tAlpha = 1; Plato::Scalar tBeta = 1;
    TEST_THROW( (Plato::multiply_matrix_workset(tNumCells, tAlpha, tA, tB, tBeta, tC)), std::runtime_error );

    // CALL FUNCTION - B IS EMPTY
    constexpr Plato::OrdinalType tNumRows = 4;
    constexpr Plato::OrdinalType tNumCols = 4;
    tA = Plato::ScalarArray3D("Matrix A", tNumCells, tNumRows, tNumCols);
    TEST_THROW( (Plato::multiply_matrix_workset(tNumCells, tAlpha, tA, tB, tBeta, tC)), std::runtime_error );

    // CALL FUNCTION - C IS EMPTY
    tB = Plato::ScalarArray3D("Matrix B", tNumCells, tNumRows + 1, tNumCols);
    TEST_THROW( (Plato::multiply_matrix_workset(tNumCells, tAlpha, tA, tB, tBeta, tC)), std::runtime_error );

    // CALL FUNCTION - NUM COLUMNS MISMATCH IN INPUT MATRICES
    tC = Plato::ScalarArray3D("Matrix C", tNumCells, tNumRows, tNumCols + 1);
    TEST_THROW( (Plato::multiply_matrix_workset(tNumCells, tAlpha, tA, tC, tBeta, tB)), std::runtime_error );

    // CALL FUNCTION - NUM ROWS MISMATCH IN INPUT MATRICES
    TEST_THROW( (Plato::multiply_matrix_workset(tNumCells, tAlpha, tA, tB, tBeta, tC)), std::runtime_error );

    // CALL FUNCTION - NUM ROWS MISMATCH IN INPUT AND OUTPUT MATRICES
    Plato::ScalarArray3D tD("Matrix D", tNumCells, tNumRows, tNumCols);
    TEST_THROW( (Plato::multiply_matrix_workset(tNumCells, tAlpha, tA, tD, tBeta, tB)), std::runtime_error );

    // CALL FUNCTION - NUM COLUMNS MISMATCH IN INPUT AND OUTPUT MATRICES
    TEST_THROW( (Plato::multiply_matrix_workset(tNumCells, tAlpha, tA, tD, tBeta, tC)), std::runtime_error );

    // CALL FUNCTION - NUM CELLS MISMATCH IN A
    Plato::ScalarArray3D tE("Matrix E", tNumCells, tNumRows, tNumCols);
    TEST_THROW( (Plato::multiply_matrix_workset(tNumCells + 1, tAlpha, tA, tD, tBeta, tE)), std::runtime_error );

    // CALL FUNCTION - NUM CELLS MISMATCH IN F
    Plato::ScalarArray3D tF("Matrix F", tNumCells + 1, tNumRows, tNumCols);
    TEST_THROW( (Plato::multiply_matrix_workset(tNumCells, tAlpha, tA, tF, tBeta, tE)), std::runtime_error );

    // CALL FUNCTION - NUM CELLS MISMATCH IN E
    Plato::ScalarArray3D tG("Matrix G", tNumCells + 1, tNumRows, tNumCols);
    TEST_THROW( (Plato::multiply_matrix_workset(tNumCells, tAlpha, tA, tD, tBeta, tG)), std::runtime_error );
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, ElastoPlasticity_MultiplyMatrixWorkset)
{
    // PREPARE DATA FOR TEST ONE
    constexpr Plato::OrdinalType tNumRows = 4;
    constexpr Plato::OrdinalType tNumCols = 4;
    constexpr Plato::OrdinalType tNumCells = 3;
    Plato::ScalarArray3D tA("Matrix A WS", tNumCells, tNumRows, tNumCols);
    Plato::Scalar tAlpha = 2;
    TEST_NOTHROW( (Plato::fill_3D_workset<tNumRows, tNumCols>(tNumCells, tAlpha, tA)) );
    Plato::ScalarArray3D tB("Matrix B WS", tNumCells, tNumRows, tNumCols);
    tAlpha = 1;
    TEST_NOTHROW( (Plato::fill_3D_workset<tNumRows, tNumCols>(tNumCells, tAlpha, tB)) );
    Plato::ScalarArray3D tC("Matrix C WS", tNumCells, tNumRows, tNumCols);
    tAlpha = 3;
    TEST_NOTHROW( (Plato::fill_3D_workset<tNumRows, tNumCols>(tNumCells, tAlpha, tC)) );

    // CALL FUNCTION
    Plato::Scalar tBeta = 1;
    TEST_NOTHROW( (Plato::multiply_matrix_workset(tNumCells, tAlpha, tA, tB, tBeta, tC)) );

    // TEST RESULTS
    constexpr Plato::Scalar tGold = 27.0;
    constexpr Plato::Scalar tTolerance = 1e-4;
    auto tHostC = Kokkos::create_mirror(tC);
    Kokkos::deep_copy(tHostC, tC);
    for(Plato::OrdinalType tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
        for(Plato::OrdinalType tRowIndex = 0; tRowIndex < tNumRows; tRowIndex++)
        {
            for(Plato::OrdinalType tColIndex = 0; tColIndex < tNumCols; tColIndex++)
            {
                //printf("(%d,%d,%d) = %f\n", tCellIndex, tRowIndex, tColIndex, tHostC(tCellIndex, tRowIndex, tColIndex));
                TEST_FLOATING_EQUALITY(tHostC(tCellIndex, tRowIndex, tColIndex), tGold, tTolerance);
            }
        }
    }

    // PREPARE DATA FOR TEST TWO
    constexpr Plato::OrdinalType tNumRows2 = 3;
    constexpr Plato::OrdinalType tNumCols2 = 3;
    Plato::ScalarArray3D tD("Matrix D WS", tNumCells, tNumRows2, tNumCols2);
    Plato::ScalarArray3D tE("Matrix E WS", tNumCells, tNumRows2, tNumCols2);
    Plato::ScalarArray3D tF("Matrix F WS", tNumCells, tNumRows2, tNumCols2);
    std::vector<std::vector<Plato::Scalar>> tData = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    auto tHostD = Kokkos::create_mirror(tD);
    auto tHostE = Kokkos::create_mirror(tE);
    auto tHostF = Kokkos::create_mirror(tF);
    for(Plato::OrdinalType tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
        for(Plato::OrdinalType tRowIndex = 0; tRowIndex < tNumRows2; tRowIndex++)
        {
            for(Plato::OrdinalType tColIndex = 0; tColIndex < tNumCols2; tColIndex++)
            {
                tHostD(tCellIndex, tRowIndex, tColIndex) = tData[tRowIndex][tColIndex];
                tHostE(tCellIndex, tRowIndex, tColIndex) = tData[tRowIndex][tColIndex];
                tHostF(tCellIndex, tRowIndex, tColIndex) = tData[tRowIndex][tColIndex];
            }
        }
    }
    Kokkos::deep_copy(tD, tHostD);
    Kokkos::deep_copy(tE, tHostE);
    Kokkos::deep_copy(tF, tHostF);

    // CALL FUNCTION - NO TRANSPOSE
    tAlpha = 1.5; tBeta = 2.5;
    TEST_NOTHROW( (Plato::multiply_matrix_workset(tNumCells, tAlpha, tD, tE, tBeta, tF)) );

    // 2. TEST RESULTS
    std::vector<std::vector<Plato::Scalar>> tGoldOut = { {47.5, 59, 70.5}, {109, 134, 159}, {170.5, 209, 247.5} };
    tHostF = Kokkos::create_mirror(tF);
    Kokkos::deep_copy(tHostF, tF);
    for(Plato::OrdinalType tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
        for(Plato::OrdinalType tRowIndex = 0; tRowIndex < tNumRows2; tRowIndex++)
        {
            for(Plato::OrdinalType tColIndex = 0; tColIndex < tNumCols2; tColIndex++)
            {
                //printf("Result(%d,%d,%d) = %f\n", tCellIndex, tRowIndex, tColIndex, tHostF(tCellIndex, tRowIndex, tColIndex));
                TEST_FLOATING_EQUALITY(tHostF(tCellIndex, tRowIndex, tColIndex), tGoldOut[tRowIndex][tColIndex], tTolerance);
            }
        }
    }
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, ElastoPlasticity_MatrixTimesVectorWorkset_Error)
{
    // PREPARE DATA
    Plato::ScalarArray3D tA;
    Plato::ScalarMultiVector tX;
    Plato::ScalarMultiVector tY;

    // CALL FUNCTION - MATRIX A IS EMPTY
    constexpr Plato::OrdinalType tNumCells = 3;
    Plato::Scalar tAlpha = 1.5; Plato::Scalar tBeta = 2.5;
    TEST_THROW( (Plato::matrix_times_vector_workset("N", tNumCells, tAlpha, tA, tX, tBeta, tY)), std::runtime_error );

    // CALL FUNCTION - VECTOR X IS EMPTY
    constexpr Plato::OrdinalType tNumCols = 2;
    constexpr Plato::OrdinalType tNumRows = 3;
    tA = Plato::ScalarArray3D("A Matrix WS", tNumCells, tNumRows, tNumCols);
    TEST_THROW( (Plato::matrix_times_vector_workset("N", tNumCells, tAlpha, tA, tX, tBeta, tY)), std::runtime_error );

    // CALL FUNCTION - VECTOR Y IS EMPTY
    tX = Plato::ScalarMultiVector("X Vector WS", tNumCells, tNumCols);
    TEST_THROW( (Plato::matrix_times_vector_workset("N", tNumCells, tAlpha, tA, tX, tBeta, tY)), std::runtime_error );

    // CALL FUNCTION - NUM CELL MISMATCH IN INPUT MATRIX
    tY = Plato::ScalarMultiVector("Y Vector WS", tNumCells, tNumRows);
    TEST_THROW( (Plato::matrix_times_vector_workset("N", tNumCells + 1, tAlpha, tA, tX, tBeta, tY)), std::runtime_error );

    // CALL FUNCTION - NUM CELL MISMATCH IN INPUT VECTOR X
    Plato::ScalarMultiVector tVecX("X Vector WS", tNumCells + 1, tNumRows);
    TEST_THROW( (Plato::matrix_times_vector_workset("N", tNumCells, tAlpha, tA, tVecX, tBeta, tY)), std::runtime_error );

    // CALL FUNCTION - NUM CELL MISMATCH IN INPUT VECTOR Y
    Plato::ScalarMultiVector tVecY("Y Vector WS", tNumCells + 1, tNumRows);
    TEST_THROW( (Plato::matrix_times_vector_workset("N", tNumCells, tAlpha, tA, tX, tBeta, tVecY)), std::runtime_error );
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, ElastoPlasticity_MatrixTimesVectorWorkset)
{
    // 1. PREPARE DATA FOR TEST ONE
    constexpr Plato::OrdinalType tNumRows = 3;
    constexpr Plato::OrdinalType tNumCols = 2;
    constexpr Plato::OrdinalType tNumCells = 3;

    // 1.1 PREPARE MATRIX DATA
    Plato::ScalarArray3D tA("A Matrix WS", tNumCells, tNumRows, tNumCols);
    std::vector<std::vector<Plato::Scalar>> tMatrixData = {{1, 2}, {3, 4}, {5, 6}};
    auto tHostA = Kokkos::create_mirror(tA);
    for(Plato::OrdinalType tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
        for(Plato::OrdinalType tRowIndex = 0; tRowIndex < tNumRows; tRowIndex++)
        {
            for(Plato::OrdinalType tColIndex = 0; tColIndex < tNumCols; tColIndex++)
            {
                tHostA(tCellIndex, tRowIndex, tColIndex) =
                        static_cast<Plato::Scalar>(tCellIndex + 1) * tMatrixData[tRowIndex][tColIndex];
            }
        }
    }
    Kokkos::deep_copy(tA, tHostA);

    // 1.2 PREPARE X VECTOR DATA
    Plato::ScalarMultiVector tX("X Vector WS", tNumCells, tNumCols);
    std::vector<Plato::Scalar> tXdata = {1, 2};
    auto tHostX = Kokkos::create_mirror(tX);
    for(Plato::OrdinalType tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
        for(Plato::OrdinalType tColIndex = 0; tColIndex < tNumCols; tColIndex++)
        {
            tHostX(tCellIndex, tColIndex) = static_cast<Plato::Scalar>(tCellIndex + 1) * tXdata[tColIndex];
        }
    }
    Kokkos::deep_copy(tX, tHostX);

    // 1.3 PREPARE Y VECTOR DATA
    Plato::ScalarMultiVector tY("Y Vector WS", tNumCells, tNumRows);
    std::vector<Plato::Scalar> tYdata = {1, 2, 3};
    auto tHostY = Kokkos::create_mirror(tY);
    for(Plato::OrdinalType tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
        for(Plato::OrdinalType tRowIndex = 0; tRowIndex < tNumRows; tRowIndex++)
        {
            tHostY(tCellIndex, tRowIndex) = static_cast<Plato::Scalar>(tCellIndex + 1) * tYdata[tRowIndex];
        }
    }
    Kokkos::deep_copy(tY, tHostY);

    // 1.4 CALL FUNCTION - NO TRANSPOSE
    Plato::Scalar tAlpha = 1.5; Plato::Scalar tBeta = 2.5;
    TEST_NOTHROW( (Plato::matrix_times_vector_workset("N", tNumCells, tAlpha, tA, tX, tBeta, tY)) );

    // 1.5 TEST RESULTS
    tHostY = Kokkos::create_mirror(tY);
    Kokkos::deep_copy(tHostY, tY);
    constexpr Plato::Scalar tTolerance = 1e-4;
    std::vector<std::vector<Plato::Scalar>> tGoldOne = { {10, 21.5, 33}, {35, 76, 117}, {75, 163.5, 252} };
    for(Plato::OrdinalType tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
        for(Plato::OrdinalType tRowIndex = 0; tRowIndex < tNumRows; tRowIndex++)
        {
            //printf("(%d,%d) = %f\n", tCellIndex, tRowIndex, tHostY(tCellIndex, tRowIndex));
            TEST_FLOATING_EQUALITY(tHostY(tCellIndex, tRowIndex), tGoldOne[tCellIndex][tRowIndex], tTolerance);
        }
    }

    // 2.1 PREPARE DATA FOR X VECTOR - TEST TWO
    Plato::ScalarMultiVector tVecX("X Vector WS", tNumCells, tNumRows);
    std::vector<Plato::Scalar> tVecXdata = {1, 2, 3};
    auto tHostVecX = Kokkos::create_mirror(tVecX);
    for(Plato::OrdinalType tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
        for(Plato::OrdinalType tRowIndex = 0; tRowIndex < tNumRows; tRowIndex++)
        {
            tHostVecX(tCellIndex, tRowIndex) = static_cast<Plato::Scalar>(tCellIndex + 1) * tVecXdata[tRowIndex];
        }
    }
    Kokkos::deep_copy(tVecX, tHostVecX);

    // 2.2 PREPARE Y VECTOR DATA
    Plato::ScalarMultiVector tVecY("Y Vector WS", tNumCells, tNumCols);
    std::vector<Plato::Scalar> tVecYdata = {1, 2};
    auto tHostVecY = Kokkos::create_mirror(tVecY);
    for(Plato::OrdinalType tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
        for(Plato::OrdinalType tColIndex = 0; tColIndex < tNumCols; tColIndex++)
        {
            tHostVecY(tCellIndex, tColIndex) = static_cast<Plato::Scalar>(tCellIndex + 1) * tVecYdata[tColIndex];
        }
    }
    Kokkos::deep_copy(tVecY, tHostVecY);

    // 2.2 CALL FUNCTION - TRANSPOSE
    TEST_NOTHROW( (Plato::matrix_times_vector_workset("T", tNumCells, tAlpha, tA, tVecX, tBeta, tVecY)) );

    // 2.3 TEST RESULTS
    tHostVecY = Kokkos::create_mirror(tVecY);
    Kokkos::deep_copy(tHostVecY, tVecY);
    std::vector<std::vector<Plato::Scalar>> tGoldTwo = { {35.5, 47}, {137, 178}, {304.5, 393} };
    for(Plato::OrdinalType tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
        for(Plato::OrdinalType tColIndex = 0; tColIndex < tNumCols; tColIndex++)
        {
            //printf("(%d,%d) = %f\n", tCellIndex, tColIndex, tHostVecY(tCellIndex, tColIndex));
            TEST_FLOATING_EQUALITY(tHostVecY(tCellIndex, tColIndex), tGoldTwo[tCellIndex][tColIndex], tTolerance);
        }
    }

    // 3. TEST VALIDITY OF TRANSPOSE
    TEST_THROW( (Plato::matrix_times_vector_workset("C", tNumCells, tAlpha, tA, tVecX, tBeta, tVecY)), std::runtime_error );
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, ElastoPlasticity_IdentityWorkset)
{
    // PREPARE DATA FOR TEST
    constexpr Plato::OrdinalType tNumRows = 4;
    constexpr Plato::OrdinalType tNumCols = 4;
    constexpr Plato::OrdinalType tNumCells = 3;
    Plato::ScalarArray3D tIdentity("tIdentity WS", tNumCells, tNumRows, tNumCols);

    // CALL FUNCTION
    Plato::identity_workset<tNumRows, tNumCols>(tNumCells, tIdentity);

    // TEST RESULTS
    constexpr Plato::Scalar tTolerance = 1e-4;
    std::vector<std::vector<Plato::Scalar>> tGold = { {1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1} };
    auto tHostIdentity = Kokkos::create_mirror(tIdentity);
    Kokkos::deep_copy(tHostIdentity, tIdentity);
    for(Plato::OrdinalType tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
        for(Plato::OrdinalType tRowIndex = 0; tRowIndex < tNumRows; tRowIndex++)
        {
            for(Plato::OrdinalType tColIndex = 0; tColIndex < tNumCols; tColIndex++)
            {
                TEST_FLOATING_EQUALITY(tHostIdentity(tCellIndex, tRowIndex, tColIndex), tGold[tRowIndex][tColIndex], tTolerance);
            }
        }
    }
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, ElastoPlasticity_InverseMatrixWorkset)
{
    // PREPARE DATA FOR TEST
    constexpr Plato::OrdinalType tNumRows = 2;
    constexpr Plato::OrdinalType tNumCols = 2;
    constexpr Plato::OrdinalType tNumCells = 3; // Number of matrices to invert
    Plato::ScalarArray3D tMatrix("Matrix A", tNumCells, 2, 2);
    auto tHostMatrix = Kokkos::create_mirror(tMatrix);
    for (Plato::OrdinalType tCellIndex = 0; tCellIndex < tNumCells; ++tCellIndex)
    {
        const Plato::Scalar tScaleFactor = 1.0 / (1.0 + tCellIndex);
        tHostMatrix(tCellIndex, 0, 0) = -2.0 * tScaleFactor;
        tHostMatrix(tCellIndex, 1, 0) = 1.0 * tScaleFactor;
        tHostMatrix(tCellIndex, 0, 1) = 1.5 * tScaleFactor;
        tHostMatrix(tCellIndex, 1, 1) = -0.5 * tScaleFactor;
    }
    Kokkos::deep_copy(tMatrix, tHostMatrix);

    // CALL FUNCTION
    Plato::ScalarArray3D tAInverse("A Inverse", tNumCells, 2, 2);
    Plato::inverse_matrix_workset<tNumRows, tNumCols>(tNumCells, tMatrix, tAInverse);

    constexpr Plato::Scalar tTolerance = 1e-6;
    std::vector<std::vector<Plato::Scalar> > tGoldMatrixInverse = { { 1.0, 3.0 }, { 2.0, 4.0 } };
    auto tHostAInverse = Kokkos::create_mirror(tAInverse);
    Kokkos::deep_copy(tHostAInverse, tAInverse);
    for (Plato::OrdinalType tMatrixIndex = 0; tMatrixIndex < tNumCells; tMatrixIndex++)
    {
        for (Plato::OrdinalType tRowIndex = 0; tRowIndex < tNumRows; tRowIndex++)
        {
            for (Plato::OrdinalType tColIndex = 0; tColIndex < tNumCols; tColIndex++)
            {
                //printf("Matrix %d Inverse (%d,%d) = %f\n", n, i, j, tHostAInverse(n, i, j));
                const Plato::Scalar tScaleFactor = (1.0 + tMatrixIndex);
                TEST_FLOATING_EQUALITY(tHostAInverse(tMatrixIndex, tRowIndex, tColIndex), tScaleFactor * tGoldMatrixInverse[tRowIndex][tColIndex], tTolerance);
            }
        }
    }
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, ElastoPlasticity_ApplyPenalty)
{
    // PREPARE DATA FOR TEST
    constexpr Plato::OrdinalType tNumRows = 3;
    constexpr Plato::OrdinalType tNumCols = 3;
    Plato::ScalarMultiVector tA("A: 2-D View", tNumRows, tNumCols);
    std::vector<std::vector<Plato::Scalar>> tData = { {10, 20, 30}, {35, 76, 117}, {75, 163, 252} };

    auto tHostA = Kokkos::create_mirror(tA);
    for (Plato::OrdinalType tRowIndex = 0; tRowIndex < tNumRows; ++tRowIndex)
    {
        for (Plato::OrdinalType tColIndex = 0; tColIndex < tNumCols; ++tColIndex)
        {
            tHostA(tRowIndex, tColIndex) = tData[tRowIndex][tColIndex];
        }
    }
    Kokkos::deep_copy(tA, tHostA);

    // CALL FUNCTION
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumRows), LAMBDA_EXPRESSION(const Plato::OrdinalType & aRowIndex)
    {
        Plato::apply_penalty<tNumCols>(aRowIndex, 0.5, tA);
    }, "identity workset");

    // TEST RESULTS
    constexpr Plato::Scalar tTolerance = 1e-6;
    tHostA = Kokkos::create_mirror(tA);
    Kokkos::deep_copy(tHostA, tA);
    std::vector<std::vector<Plato::Scalar>> tGold = { {5, 10, 15}, {17.5, 38, 58.5}, {37.5, 81.5, 126} };
    for (Plato::OrdinalType tRowIndex = 0; tRowIndex < tNumRows; tRowIndex++)
    {
        for (Plato::OrdinalType tColIndex = 0; tColIndex < tNumCols; tColIndex++)
        {
            TEST_FLOATING_EQUALITY(tHostA(tRowIndex, tColIndex), tGold[tRowIndex][tColIndex], tTolerance);
        }
    }
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, ElastoPlasticity_ComputeShearAndBulkModulus)
{
    const Plato::Scalar tPoisson = 0.3;
    const Plato::Scalar tElasticModulus = 1;
    auto tBulk = Plato::compute_bulk_modulus(tElasticModulus, tPoisson);
    constexpr Plato::Scalar tTolerance = 1e-6;
    TEST_FLOATING_EQUALITY(tBulk, 0.833333333333333, tTolerance);
    auto tShear = Plato::compute_shear_modulus(tElasticModulus, tPoisson);
    TEST_FLOATING_EQUALITY(tShear, 0.384615384615385, tTolerance);
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, ElastoPlasticity_StrainDivergence3D)
{
    // PREPARE DATA FOR TEST
    constexpr Plato::OrdinalType tNumCells = 3;
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tNumVoigtTerms = 6;
    Plato::ScalarVector tOutput("strain tensor divergence", tNumCells);
    Plato::ScalarMultiVector tStrainTensor("strain tensor", tNumCells, tNumVoigtTerms);
    auto tHostStrainTensor = Kokkos::create_mirror(tStrainTensor);
    for(Plato::OrdinalType tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
        tHostStrainTensor(tCellIndex, 0) = static_cast<Plato::Scalar>(1+tCellIndex) * 0.1;
        tHostStrainTensor(tCellIndex, 1) = static_cast<Plato::Scalar>(1+tCellIndex) * 0.2;
        tHostStrainTensor(tCellIndex, 2) = static_cast<Plato::Scalar>(1+tCellIndex) * 0.3;
        tHostStrainTensor(tCellIndex, 3) = static_cast<Plato::Scalar>(1+tCellIndex) * 0.4;
        tHostStrainTensor(tCellIndex, 4) = static_cast<Plato::Scalar>(1+tCellIndex) * 0.5;
        tHostStrainTensor(tCellIndex, 5) = static_cast<Plato::Scalar>(1+tCellIndex) * 0.6;
    }
    Kokkos::deep_copy(tStrainTensor, tHostStrainTensor);

    // CALL FUNCTION
    Plato::StrainDivergence<tSpaceDim> tComputeStrainDivergence;
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellIndex)
    {
        tComputeStrainDivergence(aCellIndex, tStrainTensor, tOutput);
    }, "test strain divergence functor");

    // TEST RESULTS
    constexpr Plato::Scalar tTolerance = 1e-6;
    std::vector<Plato::Scalar> tGold = {0.6, 1.2, 1.8};
    auto tHostOutput = Kokkos::create_mirror(tOutput);
    Kokkos::deep_copy(tHostOutput, tOutput);
    for (Plato::OrdinalType tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
        TEST_FLOATING_EQUALITY(tHostOutput(tCellIndex), tGold[tCellIndex], tTolerance);
    }
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, ElastoPlasticity_StrainDivergence2D)
{
    // PREPARE DATA FOR TEST
    constexpr Plato::OrdinalType tNumCells = 3;
    constexpr Plato::OrdinalType tSpaceDim = 2;
    constexpr Plato::OrdinalType tNumVoigtTerms = 3;
    Plato::ScalarVector tOutput("strain tensor divergence", tNumCells);
    Plato::ScalarMultiVector tStrainTensor("strain tensor", tNumCells, tNumVoigtTerms);
    auto tHostStrainTensor = Kokkos::create_mirror(tStrainTensor);
    for(Plato::OrdinalType tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
        tHostStrainTensor(tCellIndex, 0) = static_cast<Plato::Scalar>(1+tCellIndex) * 0.1;
        tHostStrainTensor(tCellIndex, 1) = static_cast<Plato::Scalar>(1+tCellIndex) * 0.2;
        tHostStrainTensor(tCellIndex, 2) = static_cast<Plato::Scalar>(1+tCellIndex) * 0.3;
    }
    Kokkos::deep_copy(tStrainTensor, tHostStrainTensor);

    // CALL FUNCTION
    Plato::StrainDivergence<tSpaceDim> tComputeStrainDivergence;
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellIndex)
    {
        tComputeStrainDivergence(aCellIndex, tStrainTensor, tOutput);
    }, "test strain divergence functor");

    // TEST RESULTS
    constexpr Plato::Scalar tTolerance = 1e-6;
    std::vector<Plato::Scalar> tGold = {0.3, 0.6, 0.9};
    auto tHostOutput = Kokkos::create_mirror(tOutput);
    Kokkos::deep_copy(tHostOutput, tOutput);
    for (Plato::OrdinalType tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
        TEST_FLOATING_EQUALITY(tHostOutput(tCellIndex), tGold[tCellIndex], tTolerance);
    }
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, ElastoPlasticity_StrainDivergence1D)
{
    // PREPARE DATA FOR TEST
    constexpr Plato::OrdinalType tNumCells = 3;
    constexpr Plato::OrdinalType tSpaceDim = 1;
    constexpr Plato::OrdinalType tNumVoigtTerms = 1;
    Plato::ScalarVector tOutput("strain tensor divergence", tNumCells);
    Plato::ScalarMultiVector tStrainTensor("strain tensor", tNumCells, tNumVoigtTerms);
    auto tHostStrainTensor = Kokkos::create_mirror(tStrainTensor);
    for(Plato::OrdinalType tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
        tHostStrainTensor(tCellIndex, 0) = static_cast<Plato::Scalar>(1+tCellIndex) * 0.1;
    }
    Kokkos::deep_copy(tStrainTensor, tHostStrainTensor);

    // CALL FUNCTION
    Plato::StrainDivergence<tSpaceDim> tComputeStrainDivergence;
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellIndex)
    {
        tComputeStrainDivergence(aCellIndex, tStrainTensor, tOutput);
    }, "test strain divergence functor");

    // TEST RESULTS
    constexpr Plato::Scalar tTolerance = 1e-6;
    std::vector<Plato::Scalar> tGold = {0.1, 0.2, 0.3};
    auto tHostOutput = Kokkos::create_mirror(tOutput);
    Kokkos::deep_copy(tHostOutput, tOutput);
    for (Plato::OrdinalType tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
        TEST_FLOATING_EQUALITY(tHostOutput(tCellIndex), tGold[tCellIndex], tTolerance);
    }
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, ElastoPlasticity_ComputeStabilization3D)
{
    // PREPARE DATA FOR TEST
    constexpr Plato::OrdinalType tNumCells = 3;
    constexpr Plato::OrdinalType tSpaceDim = 3;

    Plato::ScalarVector tCellVolume("volume", tNumCells);
    auto tHostCellVolume = Kokkos::create_mirror(tCellVolume);
    for(Plato::OrdinalType tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
        tHostCellVolume(tCellIndex, 0) = static_cast<Plato::Scalar>(1+tCellIndex) * 0.1;
    }
    Kokkos::deep_copy(tCellVolume, tHostCellVolume);

    Plato::ScalarMultiVector tPressureGrad("pressure gradient", tNumCells, tSpaceDim);
    auto tHostPressureGrad = Kokkos::create_mirror(tPressureGrad);
    for(Plato::OrdinalType tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
        tHostPressureGrad(tCellIndex, 0) = static_cast<Plato::Scalar>(1+tCellIndex) * 0.1;
        tHostPressureGrad(tCellIndex, 1) = static_cast<Plato::Scalar>(1+tCellIndex) * 0.2;
        tHostPressureGrad(tCellIndex, 2) = static_cast<Plato::Scalar>(1+tCellIndex) * 0.3;
    }
    Kokkos::deep_copy(tPressureGrad, tHostPressureGrad);

    Plato::ScalarMultiVector tProjectedPressureGrad("projected pressure gradient - gauss pt", tNumCells, tSpaceDim);
    auto tHostProjectedPressureGrad = Kokkos::create_mirror(tProjectedPressureGrad);
    for(Plato::OrdinalType tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
        tHostProjectedPressureGrad(tCellIndex, 0) = static_cast<Plato::Scalar>(1+tCellIndex) * 1;
        tHostProjectedPressureGrad(tCellIndex, 1) = static_cast<Plato::Scalar>(1+tCellIndex) * 2;
        tHostProjectedPressureGrad(tCellIndex, 2) = static_cast<Plato::Scalar>(1+tCellIndex) * 3;
    }
    Kokkos::deep_copy(tProjectedPressureGrad, tHostProjectedPressureGrad);

    // CALL FUNCTION
    constexpr Plato::Scalar tScaling = 0.5;
    constexpr Plato::Scalar tShearModulus = 2;
    Plato::ScalarMultiVector tStabilization("cell stabilization", tNumCells, tSpaceDim);
    Plato::ComputeStabilization<tSpaceDim> tComputeStabilization(tScaling, tShearModulus);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellIndex)
    {
        tComputeStabilization(aCellIndex, tCellVolume, tPressureGrad, tProjectedPressureGrad, tStabilization);
    }, "test compute stabilization functor");

    // TEST RESULTS
    constexpr Plato::Scalar tTolerance = 1e-6;
    auto tHostStabilization = Kokkos::create_mirror(tStabilization);
    Kokkos::deep_copy(tHostStabilization, tStabilization);
    std::vector<std::vector<Plato::Scalar>> tGold = {{-0.0255839119441290, -0.0511678238882572, -0.0767517358323859},
                                                     {-0.0812238574671431, -0.1624477149342860, -0.2436715724014290},
                                                     {-0.1596500440960990, -0.3193000881921980, -0.4789501322882970}};
    for (Plato::OrdinalType tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
        for (Plato::OrdinalType tDimIndex = 0; tDimIndex < tSpaceDim; tDimIndex++)
        {
            TEST_FLOATING_EQUALITY(tHostStabilization(tCellIndex, tDimIndex), tGold[tCellIndex][tDimIndex], tTolerance);
        }
    }
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, ElastoPlasticity_ComputeStabilization2D)
{
    // PREPARE DATA FOR TEST
    constexpr Plato::OrdinalType tNumCells = 3;
    constexpr Plato::OrdinalType tSpaceDim = 2;

    Plato::ScalarVector tCellVolume("volume", tNumCells);
    auto tHostCellVolume = Kokkos::create_mirror(tCellVolume);
    for(Plato::OrdinalType tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
        tHostCellVolume(tCellIndex, 0) = static_cast<Plato::Scalar>(1+tCellIndex) * 0.1;
    }
    Kokkos::deep_copy(tCellVolume, tHostCellVolume);

    Plato::ScalarMultiVector tPressureGrad("pressure gradient", tNumCells, tSpaceDim);
    auto tHostPressureGrad = Kokkos::create_mirror(tPressureGrad);
    for(Plato::OrdinalType tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
        tHostPressureGrad(tCellIndex, 0) = static_cast<Plato::Scalar>(1+tCellIndex) * 0.1;
        tHostPressureGrad(tCellIndex, 1) = static_cast<Plato::Scalar>(1+tCellIndex) * 0.2;
    }
    Kokkos::deep_copy(tPressureGrad, tHostPressureGrad);

    Plato::ScalarMultiVector tProjectedPressureGrad("projected pressure gradient - gauss pt", tNumCells, tSpaceDim);
    auto tHostProjectedPressureGrad = Kokkos::create_mirror(tProjectedPressureGrad);
    for(Plato::OrdinalType tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
        tHostProjectedPressureGrad(tCellIndex, 0) = static_cast<Plato::Scalar>(1+tCellIndex) * 1;
        tHostProjectedPressureGrad(tCellIndex, 1) = static_cast<Plato::Scalar>(1+tCellIndex) * 2;
    }
    Kokkos::deep_copy(tProjectedPressureGrad, tHostProjectedPressureGrad);

    // CALL FUNCTION
    constexpr Plato::Scalar tScaling = 0.5;
    constexpr Plato::Scalar tShearModulus = 2;
    Plato::ScalarMultiVector tStabilization("cell stabilization", tNumCells, tSpaceDim);
    Plato::ComputeStabilization<tSpaceDim> tComputeStabilization(tScaling, tShearModulus);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellIndex)
    {
        tComputeStabilization(aCellIndex, tCellVolume, tPressureGrad, tProjectedPressureGrad, tStabilization);
    }, "test compute stabilization functor");

    // TEST RESULTS
    constexpr Plato::Scalar tTolerance = 1e-6;
    auto tHostStabilization = Kokkos::create_mirror(tStabilization);
    Kokkos::deep_copy(tHostStabilization, tStabilization);
    std::vector<std::vector<Plato::Scalar>> tGold = {{-0.0255839119441290, -0.0511678238882572},
                                                     {-0.0812238574671431, -0.1624477149342860},
                                                     {-0.1596500440960990, -0.3193000881921980}};
    for (Plato::OrdinalType tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
        for (Plato::OrdinalType tDimIndex = 0; tDimIndex < tSpaceDim; tDimIndex++)
        {
            TEST_FLOATING_EQUALITY(tHostStabilization(tCellIndex, tDimIndex), tGold[tCellIndex][tDimIndex], tTolerance);
        }
    }
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, ElastoPlasticity_ComputeStabilization1D)
{
    // PREPARE DATA FOR TEST
    constexpr Plato::OrdinalType tNumCells = 3;
    constexpr Plato::OrdinalType tSpaceDim = 1;

    Plato::ScalarVector tCellVolume("volume", tNumCells);
    auto tHostCellVolume = Kokkos::create_mirror(tCellVolume);
    for(Plato::OrdinalType tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
        tHostCellVolume(tCellIndex, 0) = static_cast<Plato::Scalar>(1+tCellIndex) * 0.1;
    }
    Kokkos::deep_copy(tCellVolume, tHostCellVolume);

    Plato::ScalarMultiVector tPressureGrad("pressure gradient", tNumCells, tSpaceDim);
    auto tHostPressureGrad = Kokkos::create_mirror(tPressureGrad);
    for(Plato::OrdinalType tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
        tHostPressureGrad(tCellIndex, 0) = static_cast<Plato::Scalar>(1+tCellIndex) * 0.1;
    }
    Kokkos::deep_copy(tPressureGrad, tHostPressureGrad);

    Plato::ScalarMultiVector tProjectedPressureGrad("projected pressure gradient - gauss pt", tNumCells, tSpaceDim);
    auto tHostProjectedPressureGrad = Kokkos::create_mirror(tProjectedPressureGrad);
    for(Plato::OrdinalType tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
        tHostProjectedPressureGrad(tCellIndex, 0) = static_cast<Plato::Scalar>(1+tCellIndex) * 1;
    }
    Kokkos::deep_copy(tProjectedPressureGrad, tHostProjectedPressureGrad);

    // CALL FUNCTION
    constexpr Plato::Scalar tScaling = 0.5;
    constexpr Plato::Scalar tShearModulus = 2;
    Plato::ScalarMultiVector tStabilization("cell stabilization", tNumCells, tSpaceDim);
    Plato::ComputeStabilization<tSpaceDim> tComputeStabilization(tScaling, tShearModulus);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellIndex)
    {
        tComputeStabilization(aCellIndex, tCellVolume, tPressureGrad, tProjectedPressureGrad, tStabilization);
    }, "test compute stabilization functor");

    // TEST RESULTS
    constexpr Plato::Scalar tTolerance = 1e-6;
    auto tHostStabilization = Kokkos::create_mirror(tStabilization);
    Kokkos::deep_copy(tHostStabilization, tStabilization);
    std::vector<std::vector<Plato::Scalar>> tGold = {{-0.0255839119441290},
                                                     {-0.0812238574671431},
                                                     {-0.1596500440960990}};
    for (Plato::OrdinalType tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
        for (Plato::OrdinalType tDimIndex = 0; tDimIndex < tSpaceDim; tDimIndex++)
        {
            TEST_FLOATING_EQUALITY(tHostStabilization(tCellIndex, tDimIndex), tGold[tCellIndex][tDimIndex], tTolerance);
        }
    }
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, ElastoPlasticity_ElastoPlasticityResidual3D_NoPlasticity)
{
    // 1. PREPARE PROBLEM INPUS FOR TEST
    Plato::DataMap tDataMap;
    Omega_h::MeshSets tMeshSets;
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    Teuchos::RCP<Teuchos::ParameterList> tElastoPlasticityInputs =
        Teuchos::getParametersFromXmlString(
        "<ParameterList name='Plato Problem'>                                   \n"
        "  <ParameterList name='Material Model'>                                \n"
        "    <ParameterList name='Isotropic Linear Elastic'>                    \n"
        "      <Parameter  name='Poissons Ratio' type='double' value='0.3'/>    \n"
        "      <Parameter  name='Youngs Modulus' type='double' value='1.0e6'/>  \n"
        "    </ParameterList>                                                   \n"
        "    <ParameterList name='Elliptic'>                                    \n"
        "      <ParameterList name='Penalty Function'>                          \n"
        "        <Parameter name='Type' type='string' value='SIMP'/>            \n"
        "        <Parameter name='Exponent' type='double' value='3.0'/>         \n"
        "        <Parameter name='Minimum Value' type='double' value='1.0e-6'/> \n"
        "      </ParameterList>                                                 \n"
        "    </ParameterList>                                                   \n"
        "  </ParameterList>                                                     \n"
        "</ParameterList>                                                       \n"
      );

    // 2. PREPARE FUNCTION INPUTS FOR TEST
    const Plato::OrdinalType tNumNodes = tMesh->nverts();
    const Plato::OrdinalType tNumCells = tMesh->nelems();
    using PhysicsT = Plato::SimplexPlasticity<tSpaceDim>;
    using EvalType = typename Plato::Evaluation<PhysicsT>::Residual;
    Plato::WorksetBase<PhysicsT> tWorksetBase(*tMesh);

    // 2.1 SET CONFIGURATION
    Plato::ScalarArray3DT<EvalType::ConfigScalarType> tConfiguration("configuration", tNumCells, PhysicsT::mNumNodesPerCell, tSpaceDim);
    tWorksetBase.worksetConfig(tConfiguration);

    // 2.2 SET DESIGN VARIABLES
    Plato::ScalarMultiVectorT<EvalType::ControlScalarType> tDesignVariables("design variables", tNumCells, PhysicsT::mNumNodesPerCell);
    Kokkos::deep_copy(tDesignVariables, 1.0);

    // 2.3 SET GLOBAL STATE
    auto tNumDofsPerNode = PhysicsT::mNumDofsPerNode;
    Plato::ScalarVector tGlobalState("global state", tSpaceDim * tNumNodes);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumNodes), LAMBDA_EXPRESSION(const Plato::OrdinalType & aNodeOrdinal)
    {
        tGlobalState(aNodeOrdinal*tNumDofsPerNode+0) = (1e-7)*aNodeOrdinal; // disp_x
        tGlobalState(aNodeOrdinal*tNumDofsPerNode+1) = (2e-7)*aNodeOrdinal; // disp_y
        tGlobalState(aNodeOrdinal*tNumDofsPerNode+2) = (3e-7)*aNodeOrdinal; // disp_z
        tGlobalState(aNodeOrdinal*tNumDofsPerNode+3) = (5e-7)*aNodeOrdinal; // press
    }, "set global state");
    Plato::ScalarMultiVectorT<EvalType::StateScalarType> tCurrentGlobalState("current global state", tNumCells, PhysicsT::mNumDofsPerCell);
    tWorksetBase.worksetState(tGlobalState, tCurrentGlobalState);
    Plato::ScalarMultiVectorT<EvalType::PrevStateScalarType> tPrevGlobalState("previous global state", tNumCells, PhysicsT::mNumDofsPerCell);

    // 2.4 SET PROJECTED PRESSURE GRADIENT
    auto tNumNodesPerCell = PhysicsT::mNumNodesPerCell;
    Plato::ScalarMultiVectorT<EvalType::NodeStateScalarType> tProjectedPressureGrad("projected pressure grad", tNumCells, PhysicsT::mNumNodeStatePerCell);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        for(Plato::OrdinalType tNodeIndex=0; tNodeIndex< tNumNodesPerCell; tNodeIndex++)
        {
            for(Plato::OrdinalType tDimIndex=0; tDimIndex< tSpaceDim; tDimIndex++)
            {
                tProjectedPressureGrad(aCellOrdinal, tNodeIndex*tSpaceDim+tDimIndex) = (4e-7)*(tNodeIndex+1)*(tDimIndex+1)*(aCellOrdinal+1);
            }
        }
    }, "set projected pressure grad");

    // 2.5 SET LOCAL STATE
    Plato::ScalarMultiVectorT<EvalType::LocalStateScalarType> tCurrentLocalState("current local state", tNumCells, PhysicsT::mNumLocalDofsPerCell);
    Plato::ScalarMultiVectorT<EvalType::PrevLocalStateScalarType> tPrevLocalState("previous local state", tNumCells, PhysicsT::mNumLocalDofsPerCell);

    // 3. CALL FUNCTION
    Plato::ElastoPlasticityResidual<EvalType, PhysicsT> tComputeElastoPlasticity(*tMesh, tMeshSets, tDataMap, *tElastoPlasticityInputs);
    Plato::ScalarMultiVectorT<EvalType::ResultScalarType> tElastoPlasticityResidual("residual", tNumCells, PhysicsT::mNumDofsPerCell);
    tComputeElastoPlasticity.evaluate(tCurrentGlobalState, tPrevGlobalState, tCurrentLocalState, tPrevLocalState,
                                      tProjectedPressureGrad, tDesignVariables, tConfiguration, tElastoPlasticityResidual);

    // 4. GET GOLD VALUES - COMPARE AGAINST STABILIZED MECHANICS, NO PLASTICITY
    using GoldPhysicsT = Plato::SimplexStabilizedMechanics<tSpaceDim>;
    using GoldEvalType = typename Plato::Evaluation<GoldPhysicsT>::Residual;
    auto tResidualParams = tElastoPlasticityInputs->sublist("Elliptic");
    auto tPenaltyParams = tResidualParams.sublist("Penalty Function");
    Plato::StabilizedElastostaticResidual<GoldEvalType, Plato::MSIMP> tComputeStabilizedMech(*tMesh, tMeshSets, tDataMap, *tElastoPlasticityInputs, tPenaltyParams);
    Plato::ScalarMultiVectorT<GoldEvalType::ResultScalarType> tStabilizedMechResidual("residual", tNumCells, GoldPhysicsT::mNumDofsPerCell);
    tComputeStabilizedMech.evaluate(tCurrentGlobalState, tProjectedPressureGrad, tDesignVariables, tConfiguration, tStabilizedMechResidual);

    // 5. TEST RESULTS
    constexpr Plato::Scalar tTolerance = 1e-6;
    auto tHostGold = Kokkos::create_mirror(tStabilizedMechResidual);
    Kokkos::deep_copy(tHostGold, tStabilizedMechResidual);
    auto tHostElastoPlasticityResidual = Kokkos::create_mirror(tElastoPlasticityResidual);
    Kokkos::deep_copy(tHostElastoPlasticityResidual, tElastoPlasticityResidual);
    for(Plato::OrdinalType tCellIndex=0; tCellIndex < tNumCells; tCellIndex++)
    {
        for(Plato::OrdinalType tDofIndex=0; tDofIndex< PhysicsT::mNumDofsPerCell; tDofIndex++)
        {
            TEST_FLOATING_EQUALITY(tHostElastoPlasticityResidual(tCellIndex, tDofIndex), tHostGold(tCellIndex, tDofIndex), tTolerance);
        }
    }
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, ElastoPlasticity_ElastoPlasticityResidual2D_NoPlasticity)
{
    // 1. PREPARE PROBLEM INPUS FOR TEST
    Plato::DataMap tDataMap;
    Omega_h::MeshSets tMeshSets;
    constexpr Plato::OrdinalType tSpaceDim = 2;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    Teuchos::RCP<Teuchos::ParameterList> tElastoPlasticityInputs =
        Teuchos::getParametersFromXmlString(
        "<ParameterList name='Plato Problem'>                                   \n"
        "  <ParameterList name='Material Model'>                                \n"
        "    <ParameterList name='Isotropic Linear Elastic'>                    \n"
        "      <Parameter  name='Poissons Ratio' type='double' value='0.3'/>    \n"
        "      <Parameter  name='Youngs Modulus' type='double' value='1.0e6'/>  \n"
        "    </ParameterList>                                                   \n"
        "    <ParameterList name='Elliptic'>                                    \n"
        "      <ParameterList name='Penalty Function'>                          \n"
        "        <Parameter name='Type' type='string' value='SIMP'/>            \n"
        "        <Parameter name='Exponent' type='double' value='3.0'/>         \n"
        "        <Parameter name='Minimum Value' type='double' value='1.0e-6'/> \n"
        "      </ParameterList>                                                 \n"
        "    </ParameterList>                                                   \n"
        "  </ParameterList>                                                     \n"
        "</ParameterList>                                                       \n"
      );

    // 2. PREPARE FUNCTION INPUTS FOR TEST
    const Plato::OrdinalType tNumNodes = tMesh->nverts();
    const Plato::OrdinalType tNumCells = tMesh->nelems();
    using PhysicsT = Plato::SimplexPlasticity<tSpaceDim>;
    using EvalType = typename Plato::Evaluation<PhysicsT>::Residual;
    Plato::WorksetBase<PhysicsT> tWorksetBase(*tMesh);

    // 2.1 SET CONFIGURATION
    Plato::ScalarArray3DT<EvalType::ConfigScalarType> tConfiguration("configuration", tNumCells, PhysicsT::mNumNodesPerCell, tSpaceDim);
    tWorksetBase.worksetConfig(tConfiguration);

    // 2.2 SET DESIGN VARIABLES
    Plato::ScalarMultiVectorT<EvalType::ControlScalarType> tDesignVariables("design variables", tNumCells, PhysicsT::mNumNodesPerCell);
    Kokkos::deep_copy(tDesignVariables, 1.0);

    // 2.3 SET GLOBAL STATE
    auto tNumDofsPerNode = PhysicsT::mNumDofsPerNode;
    Plato::ScalarVector tGlobalState("global state", tSpaceDim * tNumNodes);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumNodes), LAMBDA_EXPRESSION(const Plato::OrdinalType & aNodeOrdinal)
    {
        tGlobalState(aNodeOrdinal*tNumDofsPerNode+0) = (1e-7)*aNodeOrdinal; // disp_x
        tGlobalState(aNodeOrdinal*tNumDofsPerNode+1) = (2e-7)*aNodeOrdinal; // disp_y
        tGlobalState(aNodeOrdinal*tNumDofsPerNode+2) = (3e-7)*aNodeOrdinal; // press
    }, "set global state");
    Plato::ScalarMultiVectorT<EvalType::StateScalarType> tCurrentGlobalState("current global state", tNumCells, PhysicsT::mNumDofsPerCell);
    tWorksetBase.worksetState(tGlobalState, tCurrentGlobalState);
    Plato::ScalarMultiVectorT<EvalType::PrevStateScalarType> tPrevGlobalState("previous global state", tNumCells, PhysicsT::mNumDofsPerCell);

    // 2.4 SET PROJECTED PRESSURE GRADIENT
    auto tNumNodesPerCell = PhysicsT::mNumNodesPerCell;
    Plato::ScalarMultiVectorT<EvalType::NodeStateScalarType> tProjectedPressureGrad("projected pressure grad", tNumCells, PhysicsT::mNumNodeStatePerCell);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        for(Plato::OrdinalType tNodeIndex=0; tNodeIndex< tNumNodesPerCell; tNodeIndex++)
        {
            for(Plato::OrdinalType tDimIndex=0; tDimIndex< tSpaceDim; tDimIndex++)
            {
                tProjectedPressureGrad(aCellOrdinal, tNodeIndex*tSpaceDim+tDimIndex) = (4e-7)*(tNodeIndex+1)*(tDimIndex+1)*(aCellOrdinal+1);
            }
        }
    }, "set projected pressure grad");

    // 2.5 SET LOCAL STATE
    Plato::ScalarMultiVectorT<EvalType::LocalStateScalarType> tCurrentLocalState("current local state", tNumCells, PhysicsT::mNumLocalDofsPerCell);
    Plato::ScalarMultiVectorT<EvalType::PrevLocalStateScalarType> tPrevLocalState("previous local state", tNumCells, PhysicsT::mNumLocalDofsPerCell);

    // 3. CALL FUNCTION
    Plato::ElastoPlasticityResidual<EvalType, PhysicsT> tComputeElastoPlasticity(*tMesh, tMeshSets, tDataMap, *tElastoPlasticityInputs);
    Plato::ScalarMultiVectorT<EvalType::ResultScalarType> tElastoPlasticityResidual("residual", tNumCells, PhysicsT::mNumDofsPerCell);
    tComputeElastoPlasticity.evaluate(tCurrentGlobalState, tPrevGlobalState, tCurrentLocalState, tPrevLocalState,
                                      tProjectedPressureGrad, tDesignVariables, tConfiguration, tElastoPlasticityResidual);

    // 4. GET GOLD VALUES - COMPARE AGAINST STABILIZED MECHANICS, NO PLASTICITY
    using GoldPhysicsT = Plato::SimplexStabilizedMechanics<tSpaceDim>;
    using GoldEvalType = typename Plato::Evaluation<GoldPhysicsT>::Residual;
    auto tResidualParams = tElastoPlasticityInputs->sublist("Elliptic");
    auto tPenaltyParams = tResidualParams.sublist("Penalty Function");
    Plato::StabilizedElastostaticResidual<GoldEvalType, Plato::MSIMP> tComputeStabilizedMech(*tMesh, tMeshSets, tDataMap, *tElastoPlasticityInputs, tPenaltyParams);
    Plato::ScalarMultiVectorT<GoldEvalType::ResultScalarType> tStabilizedMechResidual("residual", tNumCells, GoldPhysicsT::mNumDofsPerCell);
    tComputeStabilizedMech.evaluate(tCurrentGlobalState, tProjectedPressureGrad, tDesignVariables, tConfiguration, tStabilizedMechResidual);

    // 5. TEST RESULTS
    constexpr Plato::Scalar tTolerance = 1e-6;
    auto tHostGold = Kokkos::create_mirror(tStabilizedMechResidual);
    Kokkos::deep_copy(tHostGold, tStabilizedMechResidual);
    auto tHostElastoPlasticityResidual = Kokkos::create_mirror(tElastoPlasticityResidual);
    Kokkos::deep_copy(tHostElastoPlasticityResidual, tElastoPlasticityResidual);
    std::vector<std::vector<Plato::Scalar>> tGold =
        {{-0.3108974359, -0.0961538462, 0.2003656347 , 0.2147435897, -0.0224358974, -0.3967844462,  0.0961538462, 0.1185897436, 0.0297521448},
         {0.125,          0.0576923077, -0.0853066085, -0.0673076923, 0.1057692308, 5.45966e-07,  -0.0576923077, -0.1634615385, 0.0853060625}};
    for(Plato::OrdinalType tCellIndex=0; tCellIndex < tNumCells; tCellIndex++)
    {
        for(Plato::OrdinalType tDofIndex=0; tDofIndex< PhysicsT::mNumDofsPerCell; tDofIndex++)
        {
            //printf("residual(%d,%d) = %.10f\n", tCellIndex, tDofIndex, tHostElastoPlasticityResidual(tCellIndex, tDofIndex));
            TEST_FLOATING_EQUALITY(tHostElastoPlasticityResidual(tCellIndex, tDofIndex), tGold[tCellIndex][tDofIndex], tTolerance);
        }
    }
}

}
// ElastoPlasticityTest
