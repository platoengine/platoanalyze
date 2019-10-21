/*
 * ElastoPlasticityTest.cpp
 *
 *  Created on: Sep 30, 2019
 */

#include <memory>

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
#include "plato/LocalVectorFunctionInc.hpp"
#include "plato/ThermoPlasticityUtilities.hpp"
#include "plato/LinearTetCubRuleDegreeOne.hpp"

#include "ImplicitFunctors.hpp"
#include "plato/Plato_Solve.hpp"
#include "plato/ApplyConstraints.hpp"
#include "plato/ScalarFunctionBaseFactory.hpp"
#include "plato/ScalarFunctionIncBaseFactory.hpp"

#include "KokkosBatched_LU_Decl.hpp"
#include "KokkosBatched_LU_Serial_Impl.hpp"
#include "KokkosBatched_Trsm_Decl.hpp"
#include "KokkosBatched_Trsm_Serial_Impl.hpp"

#include <Kokkos_Concepts.hpp>
#include <KokkosBlas2_gemv.hpp>
#include <KokkosBlas3_gemm.hpp>
#include "KokkosKernels_SparseUtils.hpp"
#include "KokkosSparse_spgemm.hpp"
#include "KokkosSparse_spadd.hpp"
#include "KokkosSparse_CrsMatrix.hpp"
#include <KokkosKernels_IOUtils.hpp>

namespace Plato
{

template<Plato::OrdinalType NumLocalDofsPerCell, class AViewType, class BViewType>
void flatten_vector_workset(const Plato::OrdinalType& aNumCells,
                            AViewType& tInput,
                            BViewType& tOutput)
{
    if(tInput.size() <= static_cast<Plato::OrdinalType>(0))
    {
        THROWERR("\nInput Kokkos::View is empty, i.e. size <= 0.\n")
    }
    if(tOutput.size() <= static_cast<Plato::OrdinalType>(0))
    {
        THROWERR("\nOutput Kokkos::View is empty, i.e. size <= 0.\n")
    }
    if(Kokkos::Impl::is_view<AViewType>::value)
    {
        THROWERR("\nA is not a Kokkos::View.\n")
    }
    if(Kokkos::Impl::is_view<BViewType>::value)
    {
        THROWERR("\nB is not a Kokkos::View.\n")
    }
    if(aNumCells <= static_cast<Plato::OrdinalType>(0))
    {
        THROWERR("\nNumber of input cells, i.e. elements, is zero.\n");
    }

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, aNumCells),LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        Plato::OrdinalType tDofOrdinal = aCellOrdinal * NumLocalDofsPerCell;
        for (Plato::OrdinalType tColumn = 0; tColumn < NumLocalDofsPerCell; ++tColumn)
        {
          tOutput(tDofOrdinal + tColumn) = tInput(aCellOrdinal, tColumn);
        }
    }, "flatten residual vector");
}

template<class AViewType, class BViewType>
void matrix_update_workset(const Plato::OrdinalType& aNumCells,
                           typename AViewType::const_value_type& aAlpha,
                           const AViewType& aA,
                           typename BViewType::const_value_type& aBeta,
                           const BViewType& aB)
{
    if(aA.extent(1) != aB.extent(1))
    {
        THROWERR("\nDimension mismatch, number of rows do not match.\n")
    }
    if(aA.extent(2) != aB.extent(2))
    {
        THROWERR("\nDimension mismatch, number of columns do not match.\n")
    }
    if(Kokkos::Impl::is_view<AViewType>::value)
    {
        THROWERR("\nA is not a Kokkos::View.\n")
    }
    if(Kokkos::Impl::is_view<BViewType>::value)
    {
        THROWERR("\nB is not a Kokkos::View.\n")
    }
    if(aNumCells <= static_cast<Plato::OrdinalType>(0))
    {
        THROWERR("\nNumber of input cells, i.e. elements, is zero.\n");
    }

    const auto tNumRows = aA.extent(1);
    const auto tNumCols = aA.extent(2);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, aNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        auto tA = Kokkos::subview(aA, aCellOrdinal, Kokkos::ALL(), Kokkos::ALL());
        auto tB = Kokkos::subview(aB, aCellOrdinal, Kokkos::ALL(), Kokkos::ALL());
        for(Plato::OrdinalType tRowIndex = 0; tRowIndex < tNumRows; tRowIndex++)
        {
            for(Plato::OrdinalType tColIndex = 0; tColIndex < tNumCols; tColIndex++)
            {
                tB(tRowIndex, tColIndex) = aAlpha * tA(tRowIndex, tColIndex) + aBeta * tB(tRowIndex, tColIndex);
            }
        }
    }, "matrix update 3DView");
}

template<class XViewType, class YViewType>
void vector_update_workset(const Plato::OrdinalType& aNumCells,
                           typename XViewType::const_value_type& aAlpha,
                           const XViewType& aXvec,
                           typename YViewType::const_value_type& aBeta,
                           const YViewType& aYvec)
{
    if(aXvec.size() != aYvec.size())
    {
        THROWERR("\nDimension mismatch.\n")
    }
    if(Kokkos::Impl::is_view<XViewType>::value)
    {
        THROWERR("\nA is not a Kokkos::View.\n")
    }
    if(Kokkos::Impl::is_view<YViewType>::value)
    {
        THROWERR("\nB is not a Kokkos::View.\n")
    }
    if(aNumCells <= static_cast<Plato::OrdinalType>(0))
    {
        THROWERR("\nNumber of input cells, i.e. elements, is less or equal to zero.\n");
    }

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, aNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        auto tXvec = Kokkos::subview(aXvec, aCellOrdinal, Kokkos::ALL());
        auto tYvec = Kokkos::subview(aYvec, aCellOrdinal, Kokkos::ALL());
        const auto tLength = tXvec.size();
        for(Plato::OrdinalType tIndex = 0; tIndex < tLength; tIndex++)
        {
            tYvec(tIndex) = aAlpha * tXvec(tIndex) + aBeta * tYvec(tIndex);
        }
    }, "matrix update 3DView");
}

/******************************************************************************//**
 *
 * \brief Dense matrix-matrix multiplication: C = \f$ \beta*C + \alpha*op(A)*op(B)\f$.
 *
 * \tparam AViewType Input matrix, as a 3-D Kokkos::View
 * \tparam BViewType Input matrix, as a 3-D Kokkos::View
 * \tparam CViewType Output matrix, as a nonconst 3-D Kokkos::View
 *
 * \param transA [in] "N" for non-transpose, "T" for transpose, "C"
 *   for conjugate transpose.  All characters after the first are
 *   ignored.  This works just like the BLAS routines.
 * \param transB [in] "N" for non-transpose, "T" for transpose, "C"
 *   for conjugate transpose.  All characters after the first are
 *   ignored.  This works just like the BLAS routines.
 * \param alpha [in] Input coefficient of A*x
 * \param A [in] Input matrix, as a 2-D Kokkos::View
 * \param B [in] Input matrix, as a 2-D Kokkos::View
 * \param beta [in] Input coefficient of C
 * \param C [in/out] Output vector, as a nonconst 2-D Kokkos::View
 *
***********************************************************************************/
template<class AViewType, class BViewType, class CViewType>
void matrix_matrix_multiplication_workset(const char aTransA[],
                                          const char aTransB[],
                                          const Plato::OrdinalType& aNumCells,
                                          typename AViewType::const_value_type& aAlpha,
                                          const AViewType& aA,
                                          const BViewType& aB,
                                          typename CViewType::const_value_type& aBeta,
                                          const CViewType& aC)
{
    if(Kokkos::Impl::is_view<AViewType>::value)
    {
        THROWERR("\nA matrix is not a Kokkos::View.\n")
    }
    if(Kokkos::Impl::is_view<BViewType>::value)
    {
        THROWERR("\nB matrix is not a Kokkos::View.\n")
    }
    if(Kokkos::Impl::is_view<CViewType>::value)
    {
        THROWERR("\nC matrix is not a Kokkos::View.\n")
    }

    // Check validity of transpose argument
    bool tValidTransA = (aTransA[0] == 'N') || (aTransA[0] == 'n') ||
                        (aTransA[0] == 'T') || (aTransA[0] == 't') ||
                        (aTransA[0] == 'C') || (aTransA[0] == 'c');
    bool tValidTransB = (aTransB[0] == 'N') || (aTransB[0] == 'n') ||
                        (aTransB[0] == 'T') || (aTransB[0] == 't') ||
                        (aTransB[0] == 'C') || (aTransB[0] == 'c');
    if(!(tValidTransA && tValidTransB))
    {
        std::ostringstream tOuputStream;
        tOuputStream << "\ntransA[0] = '" << aTransA[0] << " transB[0] = '" << aTransB[0] << "'. "
                     << "Valid values include 'N' or 'n' (No transpose), 'T' or 't' (Transpose), "
                     "and 'C' or 'c' (Conjugate transpose).\n";
        THROWERR(tOuputStream.str())
    }

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, aNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        auto tA = Kokkos::subview(aA, aCellOrdinal, Kokkos::ALL(), Kokkos::ALL());
        auto tB = Kokkos::subview(aB, aCellOrdinal, Kokkos::ALL(), Kokkos::ALL());
        auto tC = Kokkos::subview(aC, aCellOrdinal, Kokkos::ALL(), Kokkos::ALL());
        KokkosBlas::gemm(aTransA, aTransB, aAlpha, tA, tB, aBeta, tC);
    }, "matrix matrix multiplication 3DView");
}

/******************************************************************************//**
 *
 * \brief Dense matrix-vector multiply: y = beta*y + alpha*A*x.
 *
 * \tparam AViewType Input matrix, as a 2-D Kokkos::View
 * \tparam XViewType Input vector, as a 1-D Kokkos::View
 * \tparam YViewType Output vector, as a nonconst 1-D Kokkos::View
 * \tparam AlphaCoeffType Type of input coefficient alpha
 * \tparam BetaCoeffType Type of input coefficient beta
 *
 * \param trans [in] "N" for non-transpose, "T" for transpose, "C"
 *   for conjugate transpose.  All characters after the first are
 *   ignored.  This works just like the BLAS routines.
 * \param alpha [in] Input coefficient of A*x
 * \param A [in] Input matrix, as a 2-D Kokkos::View
 * \param x [in] Input vector, as a 1-D Kokkos::View
 * \param beta [in] Input coefficient of y
 * \param y [in/out] Output vector, as a nonconst 1-D Kokkos::View
 *
************************************************************************************/
template<class AViewType, class XViewType, class YViewType>
void matrix_times_vector_workset(const char aTransA[],
                                 const Plato::OrdinalType& aNumCells,
                                 typename AViewType::const_value_type& aAlpha,
                                 const AViewType& aAmat,
                                 const XViewType& aXvec,
                                 typename YViewType::const_value_type& aBeta,
                                 const YViewType& aYvec)
{
    if(Kokkos::Impl::is_view<AViewType>::value)
    {
        THROWERR("\nA matrix is not a Kokkos::View.\n")
    }
    if(Kokkos::Impl::is_view<XViewType>::value)
    {
        THROWERR("\nX vector is not a Kokkos::View.\n")
    }
    if(Kokkos::Impl::is_view<YViewType>::value)
    {
        THROWERR("\nY vector is not a Kokkos::View.\n")
    }

    // Check validity of transpose argument
    bool tValidTransA = (aTransA[0] == 'N') || (aTransA[0] == 'n') ||
                        (aTransA[0] == 'T') || (aTransA[0] == 't') ||
                        (aTransA[0] == 'C') || (aTransA[0] == 'c');

    if(!tValidTransA)
    {
        std::ostringstream tOuputStream;
        tOuputStream << "\ntransA[0] = '" << aTransA[0] "'. "
                     << "Valid values include 'N' or 'n' (No transpose), 'T' or 't' (Transpose), "
                     "and 'C' or 'c' (Conjugate transpose).\n";
        THROWERR(tOuputStream.str())
    }

    // KOKKOS CHECKS DIMENSIONS INSIDE GEMV

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, aNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        auto tXvec = Kokkos::subview(aXvec, aCellOrdinal, Kokkos::ALL());
        auto tYvec = Kokkos::subview(aYvec, aCellOrdinal, Kokkos::ALL());
        auto tAmat = Kokkos::subview(aAmat, aCellOrdinal, Kokkos::ALL(), Kokkos::ALL());
        KokkosBlas::gemv(aTransA, aAlpha, tAmat, tXvec, aBeta, tYvec);
    }, "matrix vector multiplication 3DView");
}

template<class XViewType, class YViewType>
void vector_plus_vector_workset(const Plato::OrdinalType & aNumCells,
                                const Plato::Scalar & aAlpha,
                                const XViewType & aInput,
                                const YViewType & aOutput)
{
    assert(aInput.size() == aOutput.size());

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, aNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        auto tInput = Kokkos::subview(aInput, aCellOrdinal, Kokkos::ALL());
        auto tOutput = Kokkos::subview(aOutput, aCellOrdinal, Kokkos::ALL());
        tOutput(aCellOrdinal) += aAlpha * tInput(aCellOrdinal);
    }, "vector plus vector workset");
}

template<Plato::OrdinalType NumRowsPerCell, Plato::OrdinalType NumColumnsPerCell, typename ScalarType>
void convert_ad_types_to_scalar_types(const Plato::OrdinalType& aNumCells,
                                      const Plato::ScalarMultiVectorT<ScalarType>& aInput,
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
        THROWERR("\Output 3D array size is zero.\n");
    }

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, aNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
      for(Plato::OrdinalType tRowIndex = 0; tRowIndex < NumRowsPerCell; tRowIndex++)
      {
          for(Plato::OrdinalType tColumnIndex = 0; tColumnIndex < NumColumnsPerCell; tColumnIndex++)
          {
              aOutput(aCellOrdinal, tRowIndex, tColumnIndex) = aInput(aCellOrdinal, tRowIndex).dx(tColumnIndex);
          }
      }
    }, "convert AD types to scalar types");
}

template<Plato::OrdinalType NumRowsPerCell, Plato::OrdinalType NumColumnsPerCell>
void identity_3DView(const Plato::OrdinalType& aNumCells, Plato::ScalarArray3D& aInput)
{
    if(aInput.size() <= static_cast<Plato::OrdinalType>(0))
    {
        THROWERR("\nInput 3D array size is zero.\n")
    }
    if(aNumCells <= static_cast<Plato::OrdinalType>(0))
    {
        THROWERR("\nNumber of input cells, i.e. elements, is zero.\n")
    }

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, aNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        for(Plato::OrdinalType tRowIndex = 0; tRowIndex < NumRowsPerCell; tRowIndex++)
        {
            for(Plato::OrdinalType tColumnIndex = 0; tColumnIndex < NumColumnsPerCell; tColumnIndex++)
            {
                aInput(aCellOrdinal, tRowIndex, tColumnIndex) = tRowIndex == tColumnIndex ? 1.0 : 0.0;
            }
        }
    }, "fill matrix identity 3DView");
}

template<Plato::OrdinalType NumRowsPerCell, Plato::OrdinalType NumColumnsPerCell, class AViewType>
void inverse_matrix_workset(const Plato::OrdinalType& aNumCells, AViewType& aA, AViewType& aAinverse)
{
    if(aA.size() <= static_cast<Plato::OrdinalType>(0))
    {
        THROWERR("\nInput 3D array, i.e. matrix workset, size is zero.\n")
    }
    if(aAinverse.size() <= static_cast<Plato::OrdinalType>(0))
    {
        THROWERR("\Output 3D array, i.e. matrix workset, size is zero.\n")
    }
    if(aNumCells <= static_cast<Plato::OrdinalType>(0))
    {
        THROWERR("\nNumber of input cells, i.e. elements, is zero.\n")
    }

    Plato::identity_3DView<NumRowsPerCell, NumColumnsPerCell>(aNumCells, aAinverse);

    using namespace KokkosBatched;
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, aNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        auto tA = Kokkos::subview(aA, aCellOrdinal, Kokkos::ALL(), Kokkos::ALL());
        auto tAinv = Kokkos::subview(aAinverse, aCellOrdinal, Kokkos::ALL(), Kokkos::ALL());

        const Plato::Scalar tAlpha = 1.0;
        SerialLU<Algo::LU::Blocked>::invoke(tA);
        SerialTrsm<Side::Left,Uplo::Lower,Trans::NoTranspose,Diag::Unit   ,Algo::Trsm::Blocked>::invoke(tAlpha, tA, tAinv);
        SerialTrsm<Side::Left,Uplo::Upper,Trans::NoTranspose,Diag::NonUnit,Algo::Trsm::Blocked>::invoke(tAlpha, tA, tAinv);
    }, "compute matrix inverse 3DView");
}

/******************************************************************************//**
 * \brief Abstract vector function interface for Variational Multi-Scale (VMS)
 *   Partial Differential Equations (PDEs) with history dependent states
 * \tparam EvaluationType evaluation type use to determine automatic differentiation
 *   type for the vector function (e.g. Residual, Jacobian, GradientZ, GradientU, etc.)
 **********************************************************************************/
template<typename EvaluationType>
class AbstractVectorFunctionVMSInc
{
// Protected member data
protected:
    Omega_h::Mesh &mMesh;
    Plato::DataMap &mDataMap;
    Omega_h::MeshSets &mMeshSets;
    std::vector<std::string> mDofNames;

// Public access functions
public:
    /**************************************************************************//**
     * \brief Constructor
     * \param [in]  aMesh mesh metadata
     * \param [in]  aMeshSets mesh side-sets metadata
     * \param [in]  aDataMap output data map
     ******************************************************************************/
    explicit AbstractVectorFunctionVMSInc(Omega_h::Mesh &aMesh,
                                               Omega_h::MeshSets &aMeshSets,
                                               Plato::DataMap &aDataMap) :
        mMesh(aMesh),
        mDataMap(aDataMap),
        mMeshSets(aMeshSets)
    {
    }

    /**************************************************************************//**
     * \brief Destructor
     ******************************************************************************/
    virtual ~AbstractVectorFunctionVMSInc()
    {
    }

    /****************************************************************************//**
     * \brief Return reference to Omega_h mesh data base
     * \return mesh metadata
     ********************************************************************************/
    decltype(mMesh) getMesh() const
    {
        return (mMesh);
    }

    /****************************************************************************//**
     * \brief Return reference to Omega_h mesh sets
     * \return mesh side sets metadata
     ********************************************************************************/
    decltype(mMeshSets) getMeshSets() const
    {
        return (mMeshSets);
    }

    /****************************************************************************//**
     * \brief Evaluate the stabilized residual equation
     * \param [in] aGlobalState current global state ( i.e. state at the n-th time interval (\f$ t^{n} \f$) )
     * \param [in] aGlobalStatePrev previous global state ( i.e. state at the n-th minus one time interval (\f$ t^{n-1} \f$) )
     * \param [in] aLocalState current local state ( i.e. state at the n-th time interval (\f$ t^{n} \f$) )
     * \param [in] aLocalStatePrev previous local state ( i.e. state at the n-th minus one time interval (\f$ t^{n-1} \f$) )
     * \param [in] aPressureGrad current pressure gradient ( i.e. state at the n-th time interval (\f$ t^{n} \f$) )
     * \param [in] aControl design variables
     * \param [in/out] aResult residual evaluation
     * \param [in] aTimeStep current time step (i.e. \f$ \Delta{t}^{n} \f$), default = 0.0
     ********************************************************************************/
    virtual void
    evaluate(const Plato::ScalarMultiVectorT<typename EvaluationType::StateScalarType> &aGlobalState,
             const Plato::ScalarMultiVectorT<typename EvaluationType::PrevStateScalarType> &aGlobalStatePrev,
             const Plato::ScalarMultiVectorT<typename EvaluationType::LocalStateScalarType> &aLocalState,
             const Plato::ScalarMultiVectorT<typename EvaluationType::PrevLocalStateScalarType> &aLocalStatePrev,
             const Plato::ScalarMultiVectorT<typename EvaluationType::NodeStateScalarType> &aPressureGrad,
             const Plato::ScalarMultiVectorT<typename EvaluationType::ControlScalarType> &aControl,
             const Plato::ScalarArray3DT<typename EvaluationType::ConfigScalarType> &aConfig,
             const Plato::ScalarMultiVectorT<typename EvaluationType::ResultScalarType> &aResult,
             Plato::Scalar aTimeStep = 0.0) const = 0;
};
// class AbstractVectorFunctionVMSInc






template<Plato::OrdinalType Length, typename ControlType, typename ResultType>
DEVICE_TYPE inline void
apply_penalty(const Plato::OrdinalType aCellOrdinal, const ControlType & aPenalty, const Plato::ScalarMultiVectorT<ResultType> & aOutput)
{
    for(Plato::OrdinalType tIndex = 0; tIndex < Length; tIndex++)
    {
        aOutput(aCellOrdinal, tIndex) *= aPenalty;
    }
}

template<typename ScalarType>
inline ScalarType compute_shear_modulus(const ScalarType & aElasticModulus, const ScalarType & aPoissonRatio)
{
    ScalarType tShearModulus = aElasticModulus /
        ( static_cast<Plato::Scalar>(2) * ( static_cast<Plato::Scalar>(1) + aPoissonRatio) ) ;
    return (tShearModulus);
}

template<typename ScalarType>
inline ScalarType compute_bulk_modulus(const ScalarType & aElasticModulus, const ScalarType & aPoissonRatio)
{
    ScalarType tShearModulus = aElasticModulus /
        ( static_cast<Plato::Scalar>(3) * ( static_cast<Plato::Scalar>(1) - ( static_cast<Plato::Scalar>(2) * aPoissonRatio) ) );
    return (tShearModulus);
}






Plato::OrdinalType parse_num_newton_iterations(Teuchos::ParameterList & aParamList)
{
    Plato::OrdinalType tOutput = 2;
    if(aParamList.isSublist("Newton Iteration") == true)
    {
        tOutput = aParamList.sublist("Newton Iteration").get<int>("Number Iterations");
    }
    return (tOutput);
}

Plato::OrdinalType parse_num_time_steps(Teuchos::ParameterList & aParamList)
{
    Plato::OrdinalType tOutput = 1;
    if(aParamList.isSublist("Time Stepping") == true)
    {
        tOutput = aParamList.sublist("Time Stepping").get<Plato::OrdinalType>("Number Time Steps");
    }
    return (tOutput);
}

Plato::Scalar parse_time_step(Teuchos::ParameterList & aParamList)
{
    Plato::Scalar tOutput = 1.0;
    if(aParamList.isSublist("Time Stepping") == true)
    {
        tOutput = aParamList.sublist("Time Stepping").get<Plato::Scalar>("Time Step");
    }
    return (tOutput);
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





template<Plato::OrdinalType SpaceDim>
class DisplacementDivergence
{
public:
    template<typename ResultType, typename StrainType>
    DEVICE_TYPE inline ResultType
    operator()(const Plato::OrdinalType & aCellOrdinal, const Plato::ScalarMultiVectorT<StrainType> & aStrain);
};

template<>
template<typename ResultType, typename StrainType>
DEVICE_TYPE inline ResultType
DisplacementDivergence<3>::operator()(const Plato::OrdinalType & aCellOrdinal, const Plato::ScalarMultiVectorT<StrainType> & aStrain)
{
    ResultType tOutput = aStrain(aCellOrdinal, 0) + aStrain(aCellOrdinal, 1) + aStrain(aCellOrdinal, 2);
    return (tOutput);
}

template<>
template<typename ResultType, typename StrainType>
DEVICE_TYPE inline ResultType
DisplacementDivergence<2>::operator()(const Plato::OrdinalType & aCellOrdinal, const Plato::ScalarMultiVectorT<StrainType> & aStrain)
{
    ResultType tOutput = aStrain(aCellOrdinal, 0) + aStrain(aCellOrdinal, 1);
    return (tOutput);
}

template<>
template<typename ResultType, typename StrainType>
DEVICE_TYPE inline ResultType
DisplacementDivergence<1>::operator()(const Plato::OrdinalType & aCellOrdinal, const Plato::ScalarMultiVectorT<StrainType> & aStrain)
{
    ResultType tOutput = aStrain(aCellOrdinal, 0);
    return (tOutput);
}





template<Plato::OrdinalType SpaceDim>
class ComputeStabilization
{
private:
    Plato::Scalar mTwoOverThree;
    Plato::Scalar mPressureScaling;
    Plato::Scalar mElasticShearModulus;

public:
    explicit ComputeStabilization(const Plato::Scalar & aStabilization, const Plato::Scalar & aShearModulus) :
        mTwoOverThree(2.0/3.0),
        mPressureScaling(aStabilization),
        mElasticShearModulus(aShearModulus)
    {
    }

    ~ComputeStabilization()
    {
    }

    template<typename ConfigT, typename PressGradT, typename ProjPressGradT, typename ResultT>
    DEVICE_TYPE inline void operator()(const Plato::OrdinalType &aCellOrdinal,
                                       const Plato::ScalarVectorT<ConfigT> & aCellVolume,
                                       const Plato::ScalarMultiVectorT<PressGradT> &aPressureGrad,
                                       const Plato::ScalarMultiVectorT<ProjPressGradT> &aProjectedPressureGrad,
                                       const Plato::ScalarMultiVectorT<ResultT> &aStabilization);
};

template<>
template<typename ConfigT, typename PressGradT, typename ProjPressGradT, typename ResultT>
DEVICE_TYPE inline void
ComputeStabilization<3>::operator()(const Plato::OrdinalType & aCellOrdinal,
                                    const Plato::ScalarVectorT<ConfigT> & aCellVolume,
                                    const Plato::ScalarMultiVectorT<PressGradT> & aPressureGrad,
                                    const Plato::ScalarMultiVectorT<ProjPressGradT> & aProjectedPressureGrad,
                                    const Plato::ScalarMultiVectorT<ResultT> & aStabilization)
{
    ConfigT tTau = pow(aCellVolume(aCellOrdinal), mTwoOverThree) / (static_cast<Plato::Scalar>(2.0) * mElasticShearModulus);
    aStabilization(aCellOrdinal, 0) = mPressureScaling * tTau
        * (mPressureScaling * aPressureGrad(aCellOrdinal, 0) - aProjectedPressureGrad(aCellOrdinal, 0));

    aStabilization(aCellOrdinal, 1) = mPressureScaling * tTau
        * (mPressureScaling * aPressureGrad(aCellOrdinal, 1) - aProjectedPressureGrad(aCellOrdinal, 1));

    aStabilization(aCellOrdinal, 2) = mPressureScaling * tTau
        * (mPressureScaling * aPressureGrad(aCellOrdinal, 2) - aProjectedPressureGrad(aCellOrdinal, 2));
}

template<>
template<typename ConfigT, typename PressGradT, typename ProjPressGradT, typename ResultT>
DEVICE_TYPE inline void
ComputeStabilization<2>::operator()(const Plato::OrdinalType & aCellOrdinal,
                                    const Plato::ScalarVectorT<ConfigT> & aCellVolume,
                                    const Plato::ScalarMultiVectorT<PressGradT> & aPressureGrad,
                                    const Plato::ScalarMultiVectorT<ProjPressGradT> & aProjectedPressureGrad,
                                    const Plato::ScalarMultiVectorT<ResultT> & aStabilization)
{
    ConfigT tTau = pow(aCellVolume(aCellOrdinal), mTwoOverThree) / (static_cast<Plato::Scalar>(2.0) * mElasticShearModulus);
    aStabilization(aCellOrdinal, 0) = mPressureScaling * tTau
        * (mPressureScaling * aPressureGrad(aCellOrdinal, 0) - aProjectedPressureGrad(aCellOrdinal, 0));

    aStabilization(aCellOrdinal, 1) = mPressureScaling * tTau
        * (mPressureScaling * aPressureGrad(aCellOrdinal, 1) - aProjectedPressureGrad(aCellOrdinal, 1));
}

template<>
template<typename ConfigT, typename PressGradT, typename ProjPressGradT, typename ResultT>
DEVICE_TYPE inline void
ComputeStabilization<1>::operator()(const Plato::OrdinalType & aCellOrdinal,
                                    const Plato::ScalarVectorT<ConfigT> & aCellVolume,
                                    const Plato::ScalarMultiVectorT<PressGradT> & aPressureGrad,
                                    const Plato::ScalarMultiVectorT<ProjPressGradT> & aProjectedPressureGrad,
                                    const Plato::ScalarMultiVectorT<ResultT> & aStabilization)
{
    ConfigT tTau = pow(aCellVolume(aCellOrdinal), mTwoOverThree) / (static_cast<Plato::Scalar>(2.0) * mElasticShearModulus);
    aStabilization(aCellOrdinal, 0) = mPressureScaling * tTau
        * (mPressureScaling * aPressureGrad(aCellOrdinal, 0) - aProjectedPressureGrad(aCellOrdinal, 0));
}








/**************************************************************************//**
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
 ******************************************************************************/
template<typename EvaluationType, typename PhysicsType>
class ElastoPlasticityResidual: public Plato::AbstractVectorFunctionVMSInc<EvaluationType>
{
// Private member data
private:
    static constexpr auto mSpaceDim = EvaluationType::SpatialDim;               /*!< spatial dimensions */
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
    Plato::Scalar mElasticBulkModulus;             /*!< elastic bulk modulus */
    Plato::Scalar mElasticShearModulus;            /*!< elastic shear modulus */
    Plato::Scalar mElasticPropertiesPenaltySIMP;   /*!< SIMP penalty for elastic properties */
    Plato::Scalar mElasticPropertiesMinErsatzSIMP; /*!< SIMP min ersatz stiffness for elastic properties */

    std::vector<std::string> mPlotTable; /*!< array with output data identifiers */

    std::shared_ptr<Plato::BodyLoads<EvaluationType>> mBodyLoads;                                                /*!< body loads interface */
    std::shared_ptr<Plato::LinearTetCubRuleDegreeOne<mSpaceDim>> mCubatureRule;                                  /*!< linear cubature rule */
    std::shared_ptr<Plato::NaturalBCs<mSpaceDim, mNumMechDims, mNumDofsPerNode, mMechDofOffset>> mBoundaryLoads; /*!< boundary loads interface */

// Private access functions
private:
    /******************************************************************************//**
     * \brief initialize material, loads and output data
     * \param [in] aProblemParams input XML data
     **********************************************************************************/
    void initialize(Teuchos::ParameterList &aProblemParams)
    {
        auto tMaterialParamList = aProblemParams.get<Teuchos::ParameterList>("Material Model");
        this->parseIsotropicElasticMaterialProperties(tMaterialParamList);
        this->parseExternalForces(aProblemParams);

        auto tResidualParams = aProblemParams.sublist("Elliptic");
        if (tResidualParams.isType < Teuchos::Array < std::string >> ("Plottable"))
        {
            mPlotTable = tResidualParams.get < Teuchos::Array < std::string >> ("Plottable").toVector();
        }
    }

    /**************************************************************************//**
    * \brief Parse external forces
    * \param [in] aProblemParams Teuchos parameter list
    ******************************************************************************/
    void parseExternalForces(Teuchos::ParameterList &aProblemParams)
    {
        // Parse body loads
        if (aProblemParams.isSublist("Body Loads"))
        {
            mBodyLoads = std::make_shared<Plato::BodyLoads<EvaluationType>>(aProblemParams.sublist("Body Loads"));
        }

        // Parse Neumann conditions
        if (aProblemParams.isSublist("Mechanical Natural Boundary Conditions"))
        {
            mBoundaryLoads = std::make_shared<Plato::NaturalBCs<mSpaceDim, mNumMechDims, mNumDofsPerNode, mMechDofOffset>>
                (aProblemParams.sublist("Mechanical Natural Boundary Conditions"));
        }
    }

    /**************************************************************************//**
    * \brief Parse isotropic material parameters
    * \param [in] aProblemParams Teuchos parameter list
    ******************************************************************************/
    void parseIsotropicElasticMaterialProperties(Teuchos::ParameterList &aMaterialParamList)
    {
        if (aMaterialParamList.isSublist("Isotropic Linear Elastic"))
        {
            auto tElasticSubList = aMaterialParamList.sublist("Isotropic Linear Elastic");
            mPoissonsRatio = Plato::parse_poissons_ratio(tElasticSubList);
            mElasticModulus = Plato::parse_elastic_modulus(tElasticSubList);
            mElasticBulkModulus = Plato::compute_bulk_modulus(mElasticModulus, tPoissonsRatio);
            mElasticShearModulus = Plato::compute_shear_modulus(mElasticModulus, tPoissonsRatio);
            this->parsePressureTermScaling(aMaterialParamList)
        }
        else
        {
            THROWERR("'Isotropic Linear Elastic' sublist of 'Material Model' is not define.")
        }
    }

    /**************************************************************************//**
    * \brief Parse pressure scaling, needed to minimize the linear system's condition number.
    * \param [in] aProblemParams Teuchos parameter list
    ******************************************************************************/
    void parsePressureTermScaling(Teuchos::ParameterList & aMaterialParamList)
    {
        if (paramList.isType<Plato::Scalar>("Pressure Scaling"))
        {
            mPressureScaling = aMaterialParamList.get<Plato::Scalar>("Pressure Scaling");
        }
        else
        {
            mPressureScaling = mElasticBulkModulus;
        }
    }

    /**************************************************************************//**
    * \brief Copy data to output data map
    * \tparam DataT data type
    * \param [in] aData output data
    * \param [in] aName output data name
    ******************************************************************************/
    template<typename DataT>
    void outputData(const DataT & aData, const std::string & aName)
    {
        if(std::count(mPlotTable.begin(), mPlotTable.end(), aName))
        {
            toMap(mDataMap, aData, aName);
        }
    }

// Public access functions
public:
    /******************************************************************************//**
     * \brief Constructor
     * \param [in] aMesh mesh metadata
     * \param [in] aMeshSets side-sets metadata
     * \param [in] aDataMap output data map
     * \param [in] aProblemParams input XML data
     * \param [in] aPenaltyParams penalty function input XML data
     **********************************************************************************/
    ElastoPlasticityResidual(Omega_h::Mesh &aMesh,
                             Omega_h::MeshSets &aMeshSets,
                             Plato::DataMap &aDataMap,
                             Teuchos::ParameterList &aProblemParams) :
        Plato::AbstractVectorFunctionVMSInc<EvaluationType>(aMesh, aMeshSets, aDataMap),
        mPoissonsRatio(-1.0),
        mElasticModulus(-1.0),
        mElasticBulkModulus(-1.0),
        mElasticShearModulus(-1.0),
        mElasticPropertiesPenaltySIMP(3),
        mElasticPropertiesMinErsatzSIMP(1e-9),
        mBodyLoads(nullptr),
        mBoundaryLoads(nullptr),
        mCubatureRule(std::make_shared<Plato::LinearTetCubRuleDegreeOne<mSpaceDim>>())
    {
        this->initialize(aProblemParams);
    }

    /******************************************************************************//**
     * \brief Destructor
     **********************************************************************************/
    virtual ~ElastoPlasticityResidual()
    {
    }

    void setIsotropicLinearElasticProperties(const Plato::Scalar & aElasticModulus, const Plato::Scalar & aPoissonsRatio)
    {
        mPoissonsRatio = aPoissonsRatio;
        mElasticModulus = aElasticModulus;
        mElasticBulkModulus = Plato::compute_bulk_modulus(mElasticModulus, mPoissonsRatio);
        mElasticShearModulus = Plato::compute_shear_modulus(mElasticModulus, mPoissonsRatio);
    }

    /****************************************************************************//**
     * \brief Add external forces to residual
     * \param [in] aGlobalState current global state ( i.e. state at the n-th time interval (\f$ t^{n} \f$) )
     * \param [in] aControl design variables
     * \param [in/out] aResult residual evaluation
     ********************************************************************************/
    void addExternalForces(const Plato::ScalarMultiVectorT<GlobalStateT> &aGlobalState,
                           const Plato::ScalarMultiVectorT<ControlT> &aControl,
                           const Plato::ScalarMultiVectorT<ResultT> &aResult)
    {
        if (mBodyLoads != nullptr)
        {
            Plato::Scalar tScale = -1.0;
            mBodyLoads->get(mMesh, aGlobalState, aControl, aResult, tScale);
        }
    }

    /****************************************************************************//**
     * \brief Evaluate the stabilized residual equation
     * \param [in] aPressureGrad current pressure gradient ( i.e. state at the n-th time interval (\f$ t^{n} \f$) )
     * \param [in] aGlobalState current global state ( i.e. state at the n-th time interval (\f$ t^{n} \f$) )
     * \param [in] aGlobalStatePrev previous global state ( i.e. state at the n-th minus one time interval (\f$ t^{n-1} \f$) )
     * \param [in] aLocalState current local state ( i.e. state at the n-th time interval (\f$ t^{n} \f$) )
     * \param [in] aLocalStatePrev previous local state ( i.e. state at the n-th minus one time interval (\f$ t^{n-1} \f$) )
     * \param [in] aControl design variables
     * \param [in/out] aResult residual evaluation
     * \param [in] aTimeStep current time step (i.e. \f$ \Delta{t}^{n} \f$), default = 0.0
     ********************************************************************************/
    void evaluate(const Plato::ScalarMultiVectorT<GlobalStateT> &aGlobalState,
                  const Plato::ScalarMultiVectorT<PrevGlobalStateT> &aGlobalStatePrev,
                  const Plato::ScalarMultiVectorT<LocalStateT> &aLocalState,
                  const Plato::ScalarMultiVectorT<PrevLocalStateT> &aLocalStatePrev,
                  const Plato::ScalarMultiVectorT<NodeStateT> &aProjectedPressureGrad,
                  const Plato::ScalarMultiVectorT<ControlT> &aControl,
                  const Plato::ScalarArray3DT<ConfigT> &aConfig,
                  const Plato::ScalarMultiVectorT<ResultT> &aResult,
                  Plato::Scalar aTimeStep = 0.0)
    {
        auto tNumCells = mMesh.nelems();

        using GradScalarT = typename Plato::fad_type_t<PhysicsType, GlobalStateT, ConfigT>;
        using ElasticStrainT = typename Plato::fad_type_t<PhysicsType, LocalStateT, ConfigT, GlobalStateT>;

        // Functors used to compute residual-related quantities
        Plato::ScalarGrad<mSpaceDim> tComputeScalarGrad;
        Plato::ComputeGradientWorkset<mSpaceDim> tComputeGradient;
        Plato::DisplacementDivergence<mSpaceDim> tComputeDispDivergence;
        Plato::ThermoPlasticityUtilities<EvaluationType::SpatialDim, PhysicsType> tPlasticityUtils;
        Plato::ComputeStabilization<mSpaceDim> tComputeStabilization(mPressureScaling, mElasticShearModulus);
        Plato::InterpolateFromNodal<mSpaceDim, mNumDofsPerNode, mPressureDofOffset> tInterpolatePressureFromNodal;
        Plato::InterpolateFromNodal<mSpaceDim, mSpaceDim, 0 /* dof offset */, mSpaceDim> tInterpolatePressGradFromNodal;

        // Residual evaulation functors
        Plato::PressureDivergence<mSpaceDim, mNumDofsPerNode> tPressureDivergence;
        Plato::StressDivergence<mSpaceDim, mNumDofsPerNode, mMechDofOffset> tStressDivergence;
        Plato::ProjectToNode<mSpaceDim, mNumDofsPerNode, mPressureDofOffset> tProjectVolumeStrain;
        Plato::FluxDivergence<mSpaceDim, mNumDofsPerNode, mPressureDofOffset> tStabilizedDivergence;
        Plato::MSIMP tPenaltyFunction(mElasticPropertiesPenaltySIMP, mElasticPropertiesMinErsatzSIMP);

        Plato::ScalarVectorT<ResultT> tPressure("L2 pressure", tNumCells);
        Plato::ScalarVectorT<ConfigT> tCellVolume("cell volume", tNumCells);
        Plato::ScalarVectorT<ResultT> tVolumeStrain("volume strain", tNumCells);
        Plato::ScalarMultiVectorT<ResultT> tStabilization("cell stabilization", tNumCells, mSpaceDim);
        Plato::ScalarMultiVectorT<GradScalarT> tPressureGrad("pressure gradient", tNumCells, mSpaceDim);
        Plato::ScalarMultiVectorT<ResultT> tDeviatoricStress("deviatoric stress", tNumCells, mNumVoigtTerms);
        Plato::ScalarMultiVectorT<ElasticStrainT> tElasticStrain("elastic strain", tNumCells, mNumVoigtTerms);
        Plato::ScalarArray3DT<ConfigT> tConfigurationGradient("configuration gradient", tNumCells, mNumNodesPerCell, mSpaceDim);
        Plato::ScalarMultiVectorT<NodeStateT> tProjectedPressureGradGP("projected pressure gradient - gauss pt", tNumCells, mSpaceDim);

        // Transfer elasticity parameters to device
        auto tPressureScaling = mPressureScaling;
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
            tPlasticityUtils.computeElasticStrain(aCellOrdinal, aGlobalState, aLocalState,
                                                  tBasisFunctions, tConfigurationGradient, tElasticStrain);

            // compute pressure gradient
            tComputeScalarGrad(aCellOrdinal, mNumDofsPerNode, mPressureDofOffset,
                               aGlobalState, tConfigurationGradient, tPressureGrad);

            // interpolate projected pressure grad, pressure, and temperature to gauss point
            tInterpolatePressureFromNodal(aCellOrdinal, tBasisFunctions, aGlobalState, tPressure);
            tInterpolatePressGradFromNodal(aCellOrdinal, tBasisFunctions, aProjectedPressureGrad, tProjectedPressureGradGP);

            // compute cell penalty
            ControlT tDensity = Plato::cell_density<mNumNodesPerCell>(aCellOrdinal, aControl);
            ControlT tElasticPropertiesPenalty = tPenaltyFunction(tDensity);

            // compute deviatoric stress and displacement divergence
            ControlT tPenalizedShearModulus = tElasticPropertiesPenalty * tElasticShearModulus;
            tPlasticityUtils.computeDeviatoricStress(aCellOrdinal, tElasticStrain, tPenalizedShearModulus, tDeviatoricStress);
            ResultT tDispDivergence = tComputeDispDivergence(aCellOrdinal, tElasticStrain);

            // compute volume difference
            tVolumeStrain(aCellOrdinal) = tPressureScaling * tElasticPropertiesPenalty
                * (tDispDivergence - tPressure(aCellOrdinal) / tElasticBulkModulus);
            tPressure(aCellOrdinal) *= tPressureScaling * tElasticPropertiesPenalty;

            // compute cell stabilization term
            tComputeStabilization(aCellOrdinal, tCellVolume, tPressureGrad, tProjectedPressureGradGP, tStabilization);
            Plato::apply_penalty<mSpaceDim>(aCellOrdinal, tElasticPropertiesPenalty, tStabilization);

            // compute residual
            tStressDivergence (aCellOrdinal, aResult, tDeviatoricStress, tConfigurationGradient, tCellVolume);
            tPressureDivergence (aCellOrdinal, aResult, tPressure, tConfigurationGradient, tCellVolume);
            tStabilizedDivergence(aCellOrdinal, aResult, tStabilization, tConfigurationGradient, tCellVolume, -1.0);
            tProjectVolumeStrain (aCellOrdinal, tCellVolume, tBasisFunctions, tVolumeStrain, aResult);
        }, "stabilized elasto-plastic residual");

        this->addExternalForces(aGlobalState, aControl, aResult);
        this->outputData(tDeviatoricStress, "deviatoric stress");
        this->outputData(tPressure, "pressure");
    }
};
// class StabilizedElastoPlasticResidual







namespace ElastoPlasticityFactory
{

/******************************************************************************/
struct FunctionFactory
{
    /******************************************************************************//**
     * \brief Create a PLATO local vector function  inc (i.e. local residual equations)
     * \param [in] aMesh mesh database
     * \param [in] aMeshSets side sets database
     * \param [in] aDataMap output data database
     * \param [in] aInputParams input parameters
     * \param [in] aFunctionName vector function name
     * \return shared pointer to a stabilized vector function integrated in time
    **********************************************************************************/
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

/****************************************************************************//**
 * \brief Concrete class defining the Physics Type template argument for a
 * VectorFunctionVMSInc.  A VectorFunctionVMSInc is defined by a stabilized
 * Partial Differential Equation (PDE) integrated implicitly in time.  The
 * stabilization technique is based on the Variational Multiscale (VMS) method.
 * Here, the (Inc) in VectorFunctionVMSInc denotes increment.
 *******************************************************************************/
template<Plato::OrdinalType NumSpaceDim>
class ElastoPlasticity: public Plato::SimplexPlasticity<NumSpaceDim>
{
public:
    static constexpr auto mSpaceDim = NumSpaceDim;
    typedef Plato::ElastoPlasticityFactory::FunctionFactory FunctionFactory;

    using SimplexT = Plato::SimplexPlasticity<NumSpaceDim>;
    using ProjectorT = typename Plato::Projection<NumSpaceDim, SimplexT::mNumDofsPerNode, SimplexT::mPressureDofOffset, /* numProjectionDofs=*/ 1>;
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
    const Plato::OrdinalType mNumCells; /*!< total number of cells (i.e. elements) */

    Plato::DataMap& mDataMap;  /*!< output data map */
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
    /**************************************************************************//**
     * \brief Constructor
     * \param [in] aMesh mesh data base
     * \param [in] aMeshSets mesh sets data base
     * \param [in] aDataMap problem-specific data map
     * \param [in] aParamList Teuchos parameter list with input data
     * \param [in] aVectorFuncType vector function type string
     ******************************************************************************/
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

    /**************************************************************************//**
     * \brief Destructor
    ******************************************************************************/
    ~VectorFunctionVMSInc(){ return; }

    /**************************************************************************//**
     * \brief Return total number of degrees of freedom
     ******************************************************************************/
    Plato::OrdinalType size() const
    {
        return mNumNodes * mNumGlobalDofsPerNode;
    }

    /**************************************************************************//**
     * \brief Return total number of nodes
     * \return total number of nodes
     ******************************************************************************/
    Plato::OrdinalType numNodes() const
    {
        return mNumNodes;
    }

    /**************************************************************************//**
     * \brief Return total number of cells
     * \return total number of cells
     ******************************************************************************/
    Plato::OrdinalType numCells() const
    {
        return mNumCells;
    }

    /**************************************************************************//**
    * \brief Compute the global residual vector
    * \param [in] aGlobalState global state at current time step
    * \param [in] aPrevGlobalState global state at previous time step
    * \param [in] aLocalState local state at current time step
    * \param [in] aPrevLocalState local state at previous time step
    * \param [in] aControl control parameters
    * \param [in] aNodeState pressure gradient
    * \param [in] aTimeStep time step
    * \return Global residual vector
    ******************************************************************************/
    Plato::ScalarVectorT<typename Residual::ResultScalarType>
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

    /**************************************************************************//**
    * \brief Compute Jacobian with respect to (wrt) control of the global residual
    * \param [in] aGlobalState global state at current time step
    * \param [in] aPrevGlobalState global state at previous time step
    * \param [in] aLocalState local state at current time step
    * \param [in] aPrevLocalState local state at previous time step
    * \param [in] aControl control parameters
    * \param [in] aNodeState pressure gradient
    * \param [in] aTimeStep time step
    * \return Jacobian wrt control of the global residual
    ******************************************************************************/
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

    /**************************************************************************//**
    * \brief Compute Jacobian wrt configuration of the global residual
    * \param [in] aGlobalState global state at current time step
    * \param [in] aPrevGlobalState global state at previous time step
    * \param [in] aLocalState local state at current time step
    * \param [in] aPrevLocalState local state at previous time step
    * \param [in] aControl control parameters
    * \param [in] aNodeState pressure gradient
    * \param [in] aTimeStep time step
    * \return Jacobian wrt configuration of the global residual
    ******************************************************************************/
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

    /**************************************************************************//**
    * \brief Compute Jacobian wrt current global states of the global residual
    * \param [in] aGlobalState global state at current time step
    * \param [in] aPrevGlobalState global state at previous time step
    * \param [in] aLocalState local state at current time step
    * \param [in] aPrevLocalState local state at previous time step
    * \param [in] aControl control parameters
    * \param [in] aNodeState pressure gradient
    * \param [in] aTimeStep time step
    * \return Jacobian wrt current global states of the global residual
    ******************************************************************************/
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
        Plato::convert_ad_types_to_scalar_types<mNumGlobalDofsPerCell, mNumGlobalDofsPerCell>(mNumCells, tJacobianWS, tOutputJacobian);
        return tOutputJacobian;
    }

    /**************************************************************************//**
    * \brief Compute transpose Jacobian wrt current global states of the global residual
    * \param [in] aGlobalState global state at current time step
    * \param [in] aPrevGlobalState global state at previous time step
    * \param [in] aLocalState local state at current time step
    * \param [in] aPrevLocalState local state at previous time step
    * \param [in] aControl control parameters
    * \param [in] aNodeState pressure gradient
    * \param [in] aTimeStep time step
    * \return transpose Jacobian wrt current global states of the global residual
    ******************************************************************************/
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

    /**************************************************************************//**
    * \brief Compute Jacobian wrt previous global states of the global residual
    * \param [in] aGlobalState global state at current time step
    * \param [in] aPrevGlobalState global state at previous time step
    * \param [in] aLocalState local state at current time step
    * \param [in] aPrevLocalState local state at previous time step
    * \param [in] aControl control parameters
    * \param [in] aNodeState pressure gradient
    * \param [in] aTimeStep time step
    * \return Jacobian wrt previous global states of the global residual
    ******************************************************************************/
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
        Plato::convert_ad_types_to_scalar_types<mNumGlobalDofsPerCell, mNumGlobalDofsPerCell>(mNumCells, tJacobianWS, tOutputJacobian);
        return tOutputJacobian;
    }

    /**************************************************************************//**
    * \brief Compute transpose Jacobian wrt previous global states of the global residual
    * \param [in] aGlobalState global state at current time step
    * \param [in] aPrevGlobalState global state at previous time step
    * \param [in] aLocalState local state at current time step
    * \param [in] aPrevLocalState local state at previous time step
    * \param [in] aControl control parameters
    * \param [in] aNodeState pressure gradient
    * \param [in] aTimeStep time step
    * \return transpose Jacobian wrt previous global states of the global residual
    ******************************************************************************/
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

    /**************************************************************************//**
    * \brief Compute Jacobian wrt current local state of the global residual
    * \param [in] aGlobalState global state at current time step
    * \param [in] aPrevGlobalState global state at previous time step
    * \param [in] aLocalState local state at current time step
    * \param [in] aPrevLocalState local state at previous time step
    * \param [in] aControl control parameters
    * \param [in] aNodeState pressure gradient
    * \param [in] aTimeStep time step
    * \return Jacobian wrt current local state of the global residual
    ******************************************************************************/
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
        Plato::convert_ad_types_to_scalar_types<mNumGlobalDofsPerCell, mNumLocalDofsPerCell>(mNumCells, tJacobianWS, tOutputJacobian);
        return tOutputJacobian;
    }

    /**************************************************************************//**
    * \brief Compute transpose Jacobian wrt current local states of the global residual
    * \param [in] aGlobalState global state at current time step
    * \param [in] aPrevGlobalState global state at previous time step
    * \param [in] aLocalState local state at current time step
    * \param [in] aPrevLocalState local state at previous time step
    * \param [in] aControl control parameters
    * \param [in] aNodeState pressure gradient
    * \param [in] aTimeStep time step
    * \return transpose Jacobian wrt current local states of the global residual
    ******************************************************************************/
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

    /**************************************************************************//**
    * \brief Compute Jacobian wrt previous local state of the global residual
    * \param [in] aGlobalState global state at current time step
    * \param [in] aPrevGlobalState global state at previous time step
    * \param [in] aLocalState local state at current time step
    * \param [in] aPrevLocalState local state at previous time step
    * \param [in] aControl control parameters
    * \param [in] aNodeState pressure gradient
    * \param [in] aTimeStep time step
    * \return Jacobian wrt previous local state of the global residual
    ******************************************************************************/
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
        Plato::convert_ad_types_to_scalar_types<mNumGlobalDofsPerCell, mNumLocalDofsPerCell>(mNumCells, tJacobianWS, tOutputJacobian);
        return tOutputJacobian;
    }

    /**************************************************************************//**
    * \brief Compute transpose Jacobian wrt previous local states of the global residual
    * \param [in] aGlobalState global state at current time step
    * \param [in] aPrevGlobalState global state at previous time step
    * \param [in] aLocalState local state at current time step
    * \param [in] aPrevLocalState local state at previous time step
    * \param [in] aControl control parameters
    * \param [in] aNodeState pressure gradient
    * \param [in] aTimeStep time step
    * \return transpose Jacobian wrt previous local states of the global residual
    ******************************************************************************/
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

    /**************************************************************************//**
    * \brief Compute assembled Jacobian wrt pressure gradient of the global residual
    * \param [in] aGlobalState global state at current time step
    * \param [in] aPrevGlobalState global state at previous time step
    * \param [in] aLocalState local state at current time step
    * \param [in] aPrevLocalState local state at previous time step
    * \param [in] aControl control parameters
    * \param [in] aNodeState pressure gradient
    * \param [in] aTimeStep time step
    * \return Assembled Jacobian wrt pressure gradient of the global residual
    ******************************************************************************/
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

        return (this->assembleJacobianPressGrad(tJacobianWS));
    }

    /**************************************************************************//**
    * \brief Compute assembled transpose Jacobian wrt pressure gradient of the global residual
    * \param [in] aGlobalState global state at current time step
    * \param [in] aPrevGlobalState global state at previous time step
    * \param [in] aLocalState local state at current time step
    * \param [in] aPrevLocalState local state at previous time step
    * \param [in] aControl control parameters
    * \param [in] aNodeState pressure gradient
    * \param [in] aTimeStep time step
    * \return Assembled transpose Jacobian wrt pressure gradient of the global residual
    ******************************************************************************/
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

        return (this->assembleTransposeJacobianPressGrad(tJacobianWS));
    }

// Private access functions
private:
    Teuchos::RCP<Plato::CrsMatrixType>
    assembleJacobianPressGrad(const Plato::ScalarMultiVectorT<typename JacobianPgrad::ResultScalarType>& aJacobianWS)
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
          mNumGlobalDofsPerCell,    /* (Nv x Nd) */
          mNumNodeStatePerCell,     /* (Nv x Nn) */
          tJacobianMatEntryOrdinal, /* entry ordinal functor */
          aJacobianWS,              /* source data */
          tJacobianMatEntries       /* destination */
        );

        return tAssembledJacobian;
    }

    Teuchos::RCP<Plato::CrsMatrixType>
    assembleTransposeJacobianPressGrad(const Plato::ScalarMultiVectorT<typename JacobianPgrad::ResultScalarType>& aJacobianWS)
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
          mNumGlobalDofsPerCell,    /* (Nv x Nd) */
          mNumNodeStatePerCell,     /* (Nv x Nn) */
          tJacobianMatEntryOrdinal, /* entry ordinal functor */
          aJacobianWS,              /* source data */
          tJacobianMatEntries       /* destination */
        );

        return tAssembledTransposeJacobian;
    }
};
// class VectorFunctionVMSInc








class ScalarFunctionLocalHistBase
{
public:
    virtual ~ScalarFunctionLocalHistBase(){}

    /******************************************************************************//**
     * \fn virtual std::string name() const
     * \brief Return function name
     * \return user defined function name
     **********************************************************************************/
    virtual std::string name() const = 0;

    /******************************************************************************//**
     * \fn virtual Plato::Scalar value(const Plato::ScalarVector & aGlobalStates,
     *                                 const Plato::ScalarVector & aLocalStates,
     *                                 const Plato::ScalarVector & aControl,
     *                                 Plato::Scalar aTimeStep = 0.0) const
     * \brief Return function value
     * \param [in] aGlobalStates 2D view of global states
     * \param [in] aLocalStates 2D view of local states
     * \param [in] aControl 1D view of controls, i.e. design variables
     * \param [in] aTimeStep current time step increment
     * \return function value
     **********************************************************************************/
    virtual Plato::Scalar value(const Plato::ScalarVector & aGlobalStates,
                                const Plato::ScalarVector & aLocalStates,
                                const Plato::ScalarVector & aControl,
                                Plato::Scalar aTimeStep = 0.0) const = 0;

    /******************************************************************************//**
     * \fn virtual Plato::ScalarVector gradient_z(const Plato::ScalarVector & aGlobalStates,
     *                                            const Plato::ScalarVector & aLocalStates,
     *                                            const Plato::ScalarVector & aControl,
     *                                            Plato::Scalar aTimeStep = 0.0) const
     * \brief Return partial derivative wrt design variables
     * \param [in] aGlobalStates 2D view of global states
     * \param [in] aLocalStates 2D view of local states
     * \param [in] aControl 1D view of controls, i.e. design variables
     * \param [in] aTimeStep current time step increment
     * \return assembled partial derivative wrt design variables
     **********************************************************************************/
    virtual Plato::ScalarVector gradient_z(const Plato::ScalarVector & aGlobalStates,
                                           const Plato::ScalarVector & aLocalStates,
                                           const Plato::ScalarVector & aControl,
                                           Plato::Scalar aTimeStep = 0.0) const = 0;

    /******************************************************************************//**
     * \fn virtual Plato::ScalarMultiVector gradient_u(const Plato::ScalarVector & aGlobalStates,
     *                                                 const Plato::ScalarVector & aLocalStates,
     *                                                 const Plato::ScalarVector & aControl,
     *                                                 Plato::Scalar aTimeStep = 0.0) const
     * \brief Return partial derivative wrt global states workset
     * \param [in] aGlobalStates 2D view of global states
     * \param [in] aLocalStates 2D view of local states
     * \param [in] aControl 1D view of controls, i.e. design variables
     * \param [in] aTimeStep current time step increment
     * \return partial derivative wrt global states workset
     **********************************************************************************/
    virtual Plato::ScalarMultiVector gradient_u(const Plato::ScalarVector & aGlobalStates,
                                                const Plato::ScalarVector & aLocalStates,
                                                const Plato::ScalarVector & aControl,
                                                Plato::Scalar aTimeStep = 0.0) const = 0;

    /******************************************************************************//**
     * \fn virtual Plato::ScalarMultiVector gradient_u(const Plato::ScalarVector & aGlobalStates,
     *                                            const Plato::ScalarVector & aLocalStates,
     *                                            const Plato::ScalarVector & aControl,
     *                                            Plato::Scalar aTimeStep = 0.0) const
     * \brief Return partial derivative wrt local states workset
     * \param [in] aGlobalStates 2D view of global states
     * \param [in] aLocalStates 2D view of local states
     * \param [in] aControl 1D view of controls, i.e. design variables
     * \param [in] aTimeStep current time step increment
     * \return partial derivative wrt local states workset
     **********************************************************************************/
    virtual Plato::ScalarMultiVector gradient_c(const Plato::ScalarVector & aGlobalStates,
                                                const Plato::ScalarVector & aLocalStates,
                                                const Plato::ScalarVector & aControl,
                                                Plato::Scalar aTimeStep = 0.0) const = 0;

    /******************************************************************************//**
     * \fn virtual Plato::ScalarVector gradient_x(const Plato::ScalarVector & aGlobalStates,
     *                                            const Plato::ScalarVector & aLocalStates,
     *                                            const Plato::ScalarVector & aControl,
     *                                            Plato::Scalar aTimeStep = 0.0) const
     * \brief Return assembled partial derivative wrt configurtion variables
     * \param [in] aGlobalStates 2D view of global states
     * \param [in] aLocalStates 2D view of local states
     * \param [in] aControl 1D view of controls, i.e. design variables
     * \param [in] aTimeStep current time step increment
     * \return assembled partial derivative wrt configurtion variables
     **********************************************************************************/
    virtual Plato::ScalarVector gradient_x(const Plato::ScalarVector & aGlobalStates,
                                           const Plato::ScalarVector & aLocalStates,
                                           const Plato::ScalarVector & aControl,
                                           Plato::Scalar aTimeStep = 0.0) const = 0;

    /******************************************************************************//**
     * \fn virtual void updateProblem(const Plato::ScalarMultiVector & aGlobalStates,
     *                                const Plato::ScalarMultiVector & aLocalStates,
     *                                const Plato::ScalarVector & aControl) const
     * \brief Update physics-based parameters within optimization iterations
     * \param [in] aGlobalStates 2D view of global states
     * \param [in] aLocalStates 2D view of local states
     * \param [in] aControl 1D view of controls, i.e. design variables
     **********************************************************************************/
    virtual void updateProblem(const Plato::ScalarMultiVector & aGlobalStates,
                               const Plato::ScalarMultiVector & aLocalStates,
                               const Plato::ScalarVector & aControl) const = 0;
};
// class ScalarFunctionVMSIncBase







struct StateData
{
    Plato::Scalar mTimeStep; /*!< current time step */
    Plato::OrdinalType mCurrentStepIndex; /*!< current time step index */
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












/******************************************************************************//**
 * \brief Plasticity problem manager, which is responsible for performance
 * criteria evaluations and
 * \param [in] aMesh mesh database
 * \param [in] aMeshSets side sets database
 * \param [in] aInputParams input parameters database
**********************************************************************************/
template<typename SimplexPhysics>
class PlasticityProblem : public Plato::AbstractProblem
{
// private member data
private:
    static constexpr auto mSpatialDim = SimplexPhysics::mNumSpatialDims;               /*!< spatial dimensions */
    static constexpr auto mNumNodesPerCell = SimplexPhysics::mNumNodesPerCell;         /*!< number of nodes per cell */
    static constexpr auto mPressureDofOffset = SimplexPhysics::mPressureDofOffset;     /*!< number of pressure dofs offset */
    static constexpr auto mNumGlobalDofsPerNode = SimplexPhysics::mNumDofsPerNode;     /*!< number of global degrees of freedom per node */
    static constexpr auto mNumGlobalDofsPerCell = SimplexPhysics::mNumDofsPerCell;     /*!< number of global degrees of freedom per cell (i.e. element) */
    static constexpr auto mNumLocalDofsPerCell = SimplexPhysics::mNumLocalDofsPerCell; /*!< number of local degrees of freedom per cell (i.e. element) */

    // Required
    Plato::VectorFunctionVMSInc<SimplexPhysics> mGlobalResidualEq;      /*!< global equality constraint interface */
    Plato::LocalVectorFunctionInc<SimplexPhysics> mLocalResidualEq;     /*!< local equality constraint interface */
    Plato::VectorFunctionVMS<SimplexPhysics::ProjectorT> mProjectionEq; /*!< global pressure gradient projection interface */

    // Optional
    std::shared_ptr<Plato::ScalarFunctionLocalHistBase> mObjective;  /*!< objective constraint interface */
    std::shared_ptr<Plato::ScalarFunctionLocalHistBase> mConstraint; /*!< constraint constraint interface */

    Plato::Scalar mPseudoTimeStep;             /*!< pseudo time step increment */
    Plato::Scalar mInitialNormResidual;        /*!< initial norm of global residual */
    Plato::Scalar mCurrentPseudoTimeStep;      /*!< current pseudo time step */
    Plato::Scalar mNewtonRaphsonStopTolerance; /*!< Newton-Raphson stopping tolerance */

    Plato::OrdinalType mNumPseudoTimeSteps;    /*!< maximum number of pseudo time steps */
    Plato::OrdinalType mMaxNumNewtonIter;      /*!< maximum number of Newton-Raphson iterations */

    Plato::ScalarVector mGlobalResidual;       /*!< global residual */
    Plato::ScalarVector mProjResidual;         /*!< projection residual, i.e. projected pressure gradient solve residual */
    Plato::ScalarVector mProjectedPressure;    /*!< projected pressure */
    Plato::ScalarVector mProjectionAdjoint;    /*!< projection adjoint variables */

    Plato::ScalarMultiVector mLocalStates;        /*!< local state variables */
    Plato::ScalarMultiVector mLocalAdjoint;       /*!< local adjoint variables */
    Plato::ScalarMultiVector mGlobalStates;       /*!< global state variables */
    Plato::ScalarMultiVector mGlobalAdjoint;      /*!< global adjoint variables */
    Plato::ScalarMultiVector mProjectedPressGrad; /*!< projected pressure gradient (# Time Steps, # Projected Pressure Gradient dofs) */

    Teuchos::RCP<Plato::CrsMatrixType> mProjectionJacobian;   /*!< projection residual Jacobian matrix */
    Teuchos::RCP<Plato::CrsMatrixType> mGlobalJacobian; /*!< global residual Jacobian matrix */

    Plato::ScalarVector mDirichletValues; /*!< values associated with the Dirichlet boundary conditions */
    Plato::ScalarVector mDispControlDirichletValues; /*!< values associated with the Dirichlet boundary conditions at the current pseudo time step */
    Plato::LocalOrdinalVector mDirichletDofs; /*!< list of degrees of freedom associated with the Dirichlet boundary conditions */

    Plato::WorksetBase<SimplexPhysics> mWorksetBase; /*!< assembly routine interface */
    std::shared_ptr<Plato::BlockMatrixEntryOrdinal<mSpatialDim, mNumGlobalDofsPerNode>> mGlobalJacEntryOrdinal; /*!< global Jacobian matrix entry ordinal */

// public functions
public:
    /******************************************************************************//**
     * \brief PLATO Plasticity Problem constructor
     * \param [in] aMesh mesh database
     * \param [in] aMeshSets side sets database
     * \param [in] aInputParams input parameters database
    **********************************************************************************/
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

    /******************************************************************************//**
     * \brief PLATO Plasticity Problem destructor
    **********************************************************************************/
    virtual ~PlasticityProblem(){}

    /******************************************************************************//**
     * \brief Return number of global degrees of freedom in solution.
     * \return Number of global degrees of freedom
    **********************************************************************************/
    Plato::OrdinalType getNumSolutionDofs()
    {
        return (mGlobalResidualEq.size());
    }

    /******************************************************************************//**
     * \brief Set global state variables
     * \param [in] aState 2D view of global state variables - (NumTimeSteps, TotalDofs)
    **********************************************************************************/
    void setState(const Plato::ScalarMultiVector & aState)
    {
        assert(aState.extent(0) == mGlobalStates.extent(0));
        assert(aState.extent(1) == mGlobalStates.extent(1));
        Kokkos::deep_copy(mGlobalStates, aState);
    }

    /******************************************************************************//**
     * \brief Return 2D view of global state variables - (NumTimeSteps, TotalDofs)
     * \return aState 2D view of global state variables
    **********************************************************************************/
    Plato::ScalarMultiVector getState()
    {
        return mGlobalStates;
    }

    /******************************************************************************//**
     * \brief Return 2D view of global adjoint variables - (2, TotalDofs)
     * \return 2D view of global adjoint variables
    **********************************************************************************/
    Plato::ScalarMultiVector getAdjoint()
    {
        return mGlobalAdjoint;
    }

    /******************************************************************************//**
     * \brief Apply Dirichlet constraints
     * \param [in] aMatrix Compressed Row Storage (CRS) matrix
     * \param [in] aVector 1D view of Right-Hand-Side forces
    **********************************************************************************/
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

    /******************************************************************************//**
     * \brief Fill right-hand-side vector values
    **********************************************************************************/
    void applyBoundaryLoads(const Plato::ScalarVector & aForce) { return; }

    /******************************************************************************//**
     * \brief Update physics-based parameters within optimization iterations
     * \param [in] aControl 1D container of control variables
     * \param [in] aGlobalState 2D container of global state variables
    **********************************************************************************/
    void updateProblem(const Plato::ScalarVector & aControl, const Plato::ScalarMultiVector & aGlobalState)
    {
        mObjective->updateProblem(aGlobalState, mLocalStates, aControl);
        mConstraint->updateProblem(aGlobalState, mLocalStates, aControl);
    }

    /******************************************************************************//**
     * \brief Solve system of equations
     * \param [in] aControl 1D view of control variables
     * \return 2D view of state variables
    **********************************************************************************/
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
            this->updateStateData(tStateData, true /* set entries to zero */);

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

    /******************************************************************************//**
     * \fn Plato::Scalar objectiveValue(const Plato::ScalarVector & aControl,
     *                                  const Plato::ScalarMultiVector & aGlobalState)
     * \brief Evaluate objective function and return its value
     * \param [in] aControl 1D view of control variables
     * \param [in] aGlobalState 2D view of state variables
     * \return objective function value
    **********************************************************************************/
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

    /******************************************************************************//**
     * \fn Plato::Scalar objectiveValue(const Plato::ScalarVector & aControl)
     * \brief Evaluate objective function and return its value
     * \param [in] aControl 1D view of control variables
     * \return objective function value
    **********************************************************************************/
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

    /******************************************************************************//**
     * \fn Plato::Scalar constraintValue(const Plato::ScalarVector & aControl, const Plato::ScalarMultiVector & aState)
     * \brief Evaluate constraint function and return its value
     * \param [in] aControl 1D view of control variables
     * \param [in] aGlobalState 2D view of state variables
     * \return constraint function value
    **********************************************************************************/
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

    /******************************************************************************//**
     * \fn Plato::Scalar constraintValue(const Plato::ScalarVector & aControl)
     * \brief Evaluate constraint function and return its value
     * \param [in] aControl 1D view of control variables
     * \return constraint function value
    **********************************************************************************/
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

    /******************************************************************************//**
     * \brief Evaluate objective partial derivative wrt control variables
     * \param [in] aControl 1D view of control variables
     * \return 1D view - objective partial derivative wrt control variables
    **********************************************************************************/
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

    /******************************************************************************//**
     * \brief Evaluate objective gradient wrt control variables
     * \param [in] aControl 1D view of control variables
     * \param [in] aGlobalState 2D view of global state variables
     * \return 1D view of the objective gradient wrt control variables
    **********************************************************************************/
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

    /******************************************************************************//**
     * \brief Evaluate objective partial derivative wrt configuration variables
     * \param [in] aControl 1D view of control variables
     * \return 1D view - objective partial derivative wrt configuration variables
    **********************************************************************************/
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

    /******************************************************************************//**
     * \brief Evaluate objective gradient wrt configuration variables
     * \param [in] aControl 1D view of control variables
     * \param [in] aGlobalState 2D view of global state variables
     * \return 1D view of the objective gradient wrt control variables
    **********************************************************************************/
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

    /******************************************************************************//**
     * \brief Evaluate constraint partial derivative wrt control variables
     * \param [in] aControl 1D view of control variables
     * \return 1D view - constraint partial derivative wrt control variables
    **********************************************************************************/
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

    /******************************************************************************//**
     * \brief Evaluate constraint partial derivative wrt control variables
     * \param [in] aControl 1D view of control variables
     * \param [in] aState 2D view of state variables
     * \return 1D view - constraint partial derivative wrt control variables
    **********************************************************************************/
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

    /******************************************************************************//**
     * \brief Evaluate constraint partial derivative wrt configuration variables
     * \param [in] aControl 1D view of control variables
     * \return 1D view - constraint partial derivative wrt configuration variables
    **********************************************************************************/
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

    /******************************************************************************//**
     * \brief Evaluate constraint partial derivative wrt configuration variables
     * \param [in] aControl 1D view of control variables
     * \param [in] aState 2D view of state variables
     * \return 1D view - constraint partial derivative wrt configuration variables
    **********************************************************************************/
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

    /******************************************************************************//**
     * \brief Update state data for time step n, i.e. current time step:
     * \param [in] aStateData state data manager
     * \param [in] aZeroEntries flag - zero all entries in current states (default = false)
    **********************************************************************************/
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

    /******************************************************************************//**
     * \brief Update adjoint data for time step n, i.e. current time step:
     * \param [in] aAdjointData adjoint data manager
    **********************************************************************************/
    void updateAdjointData(Plato::AdjointData& aAdjointData)
    {
        const Plato::OrdinalType tCurrentStepIndex = 1;
        aAdjointData.mCurrentLocalAdjoint = Kokkos::subview(mLocalAdjoint, tCurrentStepIndex, Kokkos::ALL());
        aAdjointData.mCurrentGlobalAdjoint = Kokkos::subview(mGlobalAdjoint, tCurrentStepIndex, Kokkos::ALL());
        Plato::Scalar tAlpha = 1.0; Plato::Scalar tBeta = 0.0;
        Plato::update(tAlpha, aAdjointData.mCurrentLocalAdjoint, tBeta, aAdjointData.mPreviousLocalAdjoint);
        Plato::update(tAlpha, aAdjointData.mCurrentGlobalAdjoint, tBeta, aAdjointData.mPreviousGlobalAdjoint);
    }

    /******************************************************************************//**
     * \brief Update inverse of local Jacobian wrt local states, i.e.
     * /f$ \left[ \left( \frac{\partial{H}}{\partial{c}} \right)_{t=n} \right]^{-1}, /f$:
     *
     * where H is the local residual and c is the local state vector. The pseudo time is
     * denoted by t, where n denotes the current step index.
     *
     * \param [in] aControl 1D view of control variables, i.e. design variables
     * \param [in] aStateData state data manager
     * \param [in] aInvLocalJacobianT inverse of local Jacobian wrt local states
    **********************************************************************************/
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

    /******************************************************************************//**
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
    **********************************************************************************/
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

    /******************************************************************************//**
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
    **********************************************************************************/
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

    /******************************************************************************//**
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
    **********************************************************************************/
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

    /******************************************************************************//**
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
    **********************************************************************************/
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
            auto tDfDc = aCriterion.gradient_c(aStateData.mCurrentGlobalState, aStateData.mCurrentLocalState, aControl, aStateData.mTimeStep);
            Plato::vector_plus_vector_workset(tNumCells, tAlpha, tDfDc, tRHS);
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

    /******************************************************************************//**
     * \brief Update global adjoint vector
     *
     * \param [in] aControl 1D view of control variables, i.e. design variables
     * \param [in] aStateData state data manager
     * \param [in] aInvLocalJacobianT inverse of transpose local Jacobian wrt local states
     * \param [in] aAdjointData adjoint data manager
     * \param [in] aCriterion performance criterion interface
    **********************************************************************************/
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

    /******************************************************************************//**
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
    **********************************************************************************/
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
            Plato::vector_plus_vector_workset(tNumCells, tAlpha, tDfDc, tWorkMultiVectorOneWS);

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
            tAlpha = -1.0;
            Plato::vector_plus_vector_workset(tNumCells, tAlpha, tRHS, tDfDu);
        }

        Plato::fill(static_cast<Plato::Scalar>(0), mGlobalResidual);
        mWorksetBase.assembleVectorGradientU(tDfDu, mGlobalResidual);
        Plato::scale(static_cast<Plato::Scalar>(-1), mGlobalResidual)
    }

    /******************************************************************************//**
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
    **********************************************************************************/
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
        Plato::Condense(mGlobalJacobian /* K_t */, tDpDn_T, mProjectionJacobian,  tDrDn_T, mPressureDofOffset /* row offset */);
    }

    /******************************************************************************//**
     * \brief Compute Schur complement, which is defined as /f$ A = \frac{\partial{R}}
     * {\partial{c}} * \left[ \left(\frac{\partial{H}}{\partial{c}}\right)^{-1} *
     * \frac{\partial{H}}{\partial{u}} \right] /f$, where R is the global residual, H
     * is the local residual, u are the global states, and c are the local states.
     *
     * \param [in] aControl 1D view of control variables, i.e. design variables
     * \param [in] aStateData state data manager
     * \param [in] aInvLocalJacobianT inverse of transpose local Jacobian wrt local states
     * \return 3D view with Schur complement per cell
    **********************************************************************************/
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
        Plato::matrix_matrix_multiplication_workset("N", "N", tNumCells, tAlpha, aInvLocalJacobianT, tDhDu, tBeta, tInvDhDcTimesDhDu);


        // Compute cell Jacobian of the global residual with respect to the current local state WorkSet (WS)
        auto tDrDc = mGlobalResidualEq.gradient_c(aStateData.mCurrentGlobalState, aStateData.mPreviousGlobalState,
                                                  aStateData.mCurrentLocalState, aStateData.mPreviousLocalState,
                                                  aStateData.mCurrentProjPressGrad, aControl, aStateData.mTimeStep);

        // Compute cell Schur = dR/dc * (dH/dc)^{-1} * dH/du, where H is the local residual,
        // R is the global residual, c are the local states and u are the global states
        Plato::ScalarArray3D tOutput("Schur Complement", tNumCells, mNumLocalDofsPerCell, mNumLocalDofsPerCell);
        Plato::matrix_matrix_multiplication_workset("N", "N", tNumCells, tAlpha, tDrDc, tInvDhDcTimesDhDu, tBeta, tOutput);
        return tOutput;
    }

    /******************************************************************************//**
     * \brief Assemble tangent stiffness matrix, which is defined as /f$ K_{T} =
     * \frac{\partial{R}}{\partial{u}} - \frac{\partial{R}}{\partial{c}} * \left[
     * \left(\frac{\partial{H}}{\partial{c}}\right)^{-1} * \frac{\partial{H}}{\partial{u}}
     * \right] /f$, where R is the global residual, H is the local residual, u are
     * the global states, and c are the local states.
     *
     * \param [in] aControl 1D view of control variables, i.e. design variables
     * \param [in] aStateData state data manager
     * \param [in] aInvLocalJacobianT inverse of transpose local Jacobian wrt local states
    **********************************************************************************/
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
        Plato::matrix_update_workset(tNumCells, tAlpha, tSchurComplement, tBeta, tDrDu);

        auto tJacobianEntries = mGlobalJacobian->entries();
        Plato::assemble_jacobian(tNumCells, mNumGlobalDofsPerCell, mNumGlobalDofsPerCell,
                                 *mGlobalJacEntryOrdinal, tDrDu, tJacobianEntries);
    }

    /******************************************************************************//**
     * \brief Set Dirichlet degrees of freedom in global state increment view to zero
     * \param [in] aGlobalStateIncrement 1D view of global state increments, i.e. Newton-Raphson solution
    **********************************************************************************/
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

    /******************************************************************************//**
     * \brief Check Newton-Raphson solver convergence criterion
     * \param [in] aNewtonIteration current Newton-Raphson iteration
     * \return boolean flag, indicates if Newton-Raphson solver converged
    **********************************************************************************/
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
    /******************************************************************************//**
     * \brief Initialize member data
     * \param [in] aMesh mesh database
     * \param [in] aMeshSets side sets database
     * \param [in] aInputParams input parameters database
     **********************************************************************************/
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

    /******************************************************************************//**
     * \brief Allocate objective function interface and adjoint containers
     * \param [in] aMesh mesh database
     * \param [in] aMeshSets side sets database
     * \param [in] aInputParams input parameters database
     **********************************************************************************/
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

    /******************************************************************************//**
     * \brief Allocate constraint function interface and adjoint containers
     * \param [in] aMesh mesh database
     * \param [in] aMeshSets side sets database
     * \param [in] aInputParams input parameters database
     **********************************************************************************/
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

}
// namespace Plato

namespace ElastoPlasticityTest
{

}
// ElastoPlasticityTest
