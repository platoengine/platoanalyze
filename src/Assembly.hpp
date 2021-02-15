#pragma once

#include "SpatialModel.hpp"

namespace Plato
{

/******************************************************************************//**
* \brief Combines values from all threads and return the combined result.
*
* \tparam Scalar output POD type
* \tparam Result input POD type
*
* \param [in] aNumCells  number of elements/cells
* \param [in] aResult    input 1-D view
*
* \return output sum
*
**********************************************************************************/
template <class Scalar, class Result>
inline Scalar local_result_sum(const Plato::OrdinalType& aNumCells, const Result & aResult)
{
  Scalar tReturnVal(0.0);
  Kokkos::parallel_reduce(Kokkos::RangePolicy<>(0,aNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal, Scalar& aLocalResult)
  {
    aLocalResult += aResult(aCellOrdinal);
  }, tReturnVal);
  return tReturnVal;
}
// function local_result_sum


/******************************************************************************//**
 * \brief Flatten vector workset. Takes 2D view and converts it into a 1D view.
 *
 * \tparam NumDofsPerCell   number of degrees of freedom per cell
 * \tparam AViewType Input  workset, as a 2-D Kokkos::View
 * \tparam BViewType Output workset, as a 1-D Kokkos::View
 *
 * \param [in]     aNumCells number of cells, i.e. elements
 * \param [in]     aInput    input workset (NumCells, LocalNumCellDofs)
 * \param [in/out] aOutput   output vector (NumCells * LocalNumCellDofs)
**********************************************************************************/
template<Plato::OrdinalType NumDofsPerCell, class AViewType, class BViewType>
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
        const auto tDofOffset = aCellOrdinal * NumDofsPerCell;
        for (Plato::OrdinalType tDofIndex = 0; tDofIndex < NumDofsPerCell; tDofIndex++)
        {
          aOutput(tDofOffset + tDofIndex) = aInput(aCellOrdinal, tDofIndex);
        }
    }, "flatten residual vector");
}
// function flatten_vector_workset

/******************************************************************************//**
 * \brief Flatten vector workset. Takes 2D view and converts it into a 1D view.
 *
 * \tparam NumDofsPerCell   number of degrees of freedom per cell
 * \tparam AViewType Input  workset, as a 2-D Kokkos::View
 * \tparam BViewType Output workset, as a 1-D Kokkos::View
 *
 * \param [in]     aNumCells number of cells, i.e. elements
 * \param [in]     aInput    input workset (NumCells, LocalNumCellDofs)
 * \param [in/out] aOutput   output vector (NumCells * LocalNumCellDofs)
**********************************************************************************/
template<Plato::OrdinalType NumDofsPerCell, class AViewType, class BViewType>
inline void
flatten_vector_workset(
    const Plato::SpatialDomain & aDomain,
          AViewType            & aInput,
          BViewType            & aOutput
)
{
    auto tNumCells = aDomain.numCells();
    auto tCellOrdinals = aDomain.cellOrdinals();

    if(aInput.size() <= static_cast<Plato::OrdinalType>(0))
    {
        THROWERR("\nInput Kokkos::View is empty, i.e. size <= 0.\n")
    }
    if(aOutput.size() <= static_cast<Plato::OrdinalType>(0))
    {
        THROWERR("\nOutput Kokkos::View is empty, i.e. size <= 0.\n")
    }
    if(tNumCells <= static_cast<Plato::OrdinalType>(0))
    {
        THROWERR("\nNumber of cells, i.e. elements, argument is <= zero.\n");
    }

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells),LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        auto tCellOrdinal = tCellOrdinals[aCellOrdinal];
        const auto tDofOffset = tCellOrdinal * NumDofsPerCell;
        for (Plato::OrdinalType tDofIndex = 0; tDofIndex < NumDofsPerCell; tDofIndex++)
        {
          aOutput(tDofOffset + tDofIndex) = aInput(aCellOrdinal, tDofIndex);
        }
    }, "flatten residual vector");
}
// function flatten_vector_workset

/******************************************************************************//**
 * \brief Flatten vector workset. Takes 2D view and converts it into a 1D view.
 *
 * \tparam NumDofsPerCell   number of degrees of freedom per cell
 * \tparam AViewType Input  workset, as a 2-D Kokkos::View
 * \tparam BViewType Output workset, as a 1-D Kokkos::View
 *
 * \param [in]     aNumCells number of cells, i.e. elements
 * \param [in]     aInput    input workset (NumCells, LocalNumCellDofs)
 * \param [in/out] aOutput   output vector (NumCells * LocalNumCellDofs)
**********************************************************************************/
template<Plato::OrdinalType NumDofsPerCell, class ViewType>
inline void
assemble_vector_workset(
    const Plato::SpatialDomain & aDomain,
          ViewType             & aInput,
          ViewType             & aOutput
)
{
    auto tNumCells = aDomain.numCells();
    auto tCellOrdinals = aDomain.cellOrdinals();

    if(aInput.size() <= static_cast<Plato::OrdinalType>(0))
    {
        THROWERR("\nInput Kokkos::View is empty, i.e. size <= 0.\n")
    }
    if(aOutput.size() <= static_cast<Plato::OrdinalType>(0))
    {
        THROWERR("\nOutput Kokkos::View is empty, i.e. size <= 0.\n")
    }
    if(tNumCells <= static_cast<Plato::OrdinalType>(0))
    {
        THROWERR("\nNumber of cells, i.e. elements, argument is <= zero.\n");
    }

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells),LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        auto tCellOrdinal = tCellOrdinals[aCellOrdinal];
        for (Plato::OrdinalType tDofIndex = 0; tDofIndex < NumDofsPerCell; tDofIndex++)
        {
            aOutput(tCellOrdinal, tDofIndex) = aInput(aCellOrdinal, tDofIndex);
        }
    }, "combine residual vector");
}
// function assemble_vector_workset

/************************************************************************//**
 *
 * \brief (2D-View) Transform automatic differentiation (AD) type to POD.
 *
 * \tparam NumNodesPerCell number of nodes per cell
 * \tparam ADType          AD scalar type
 *
 * \param [in]     aNumCells  number of cells
 * \param [in]     aInput     AD partial derivative
 * \param [in\out] aOutput    Scalar partial derivative
 *
 ********************************************************************************/
template<Plato::OrdinalType NumDofsPerCell, typename ADType>
inline void transform_ad_type_to_pod_2Dview(const Plato::ScalarVectorT<ADType>& aInput,
                                                  Plato::ScalarMultiVector& aOutput)
{
    if(aInput.extent(0) != aOutput.extent(0))
    {
        THROWERR("Dimension mismatch, input and output containers have different number of rows.")
    }
    if(NumDofsPerCell != aOutput.extent(1))
    {
        THROWERR("Input number of degrees of freedom does not match the number of columns in output container.")
    }

    Plato::OrdinalType tNumCells = aOutput.extent(0);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        for(Plato::OrdinalType tDimIndex=0; tDimIndex < NumDofsPerCell; tDimIndex++)
        {
            aOutput(aCellOrdinal, tDimIndex) = aInput(aCellOrdinal).dx(tDimIndex);
        }
    }, "Convert AD Partial to POD type");
}
// function transform_ad_type_to_pod_2Dview

/************************************************************************//**
 *
 * \brief (2D-View) Transform automatic differentiation (AD) type to POD.
 *
 * \tparam NumNodesPerCell number of nodes per cell
 * \tparam ADType          AD scalar type
 *
 * \param [in]     aNumCells  number of cells
 * \param [in]     aInput     AD partial derivative
 * \param [in\out] aOutput    Scalar partial derivative
 *
 ********************************************************************************/
template<Plato::OrdinalType NumDofsPerCell, typename ADType>
inline void
transform_ad_type_to_pod_2Dview(
    const Plato::SpatialDomain         & aDomain,
    const Plato::ScalarVectorT<ADType> & aInput,
          Plato::ScalarMultiVector     & aOutput)
{
    if(aInput.extent(0) != aDomain.numCells())
    {
        THROWERR("Dimension mismatch, input container and domain have different number of rows.")
    }
    if(NumDofsPerCell != aOutput.extent(1))
    {
        THROWERR("Input number of degrees of freedom does not match the number of columns in output container.")
    }

    auto tNumCells = aDomain.numCells();
    auto tCellOrdinals = aDomain.cellOrdinals();
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        auto tCellOrdinal = tCellOrdinals[aCellOrdinal];
        for(Plato::OrdinalType tDimIndex=0; tDimIndex < NumDofsPerCell; tDimIndex++)
        {
            aOutput(tCellOrdinal, tDimIndex) = aInput(aCellOrdinal).dx(tDimIndex);
        }
    }, "Convert AD Partial to POD type");
}
// function transform_ad_type_to_pod_2Dview

/************************************************************************//**
 *
 * \brief (3D-View) Transform automatic differentiation (AD) type to POD.
 *
 * \tparam NumRowsPerCell number of rows per cell
 * \tparam NumColsPerCell number of columns per cell
 * \tparam ADType         AD scalar type
 *
 * \param [in]     aNumCells  number of cells
 * \param [in]     aInput     AD Jacobian
 * \param [in/out] aOutput    Scalar Jacobian
 *
********************************************************************************/
template<Plato::OrdinalType NumRowsPerCell, Plato::OrdinalType NumColsPerCell, typename ADType>
inline void
transform_ad_type_to_pod_3Dview(
    const Plato::SpatialDomain              & aDomain,
    const Plato::ScalarMultiVectorT<ADType> & aInput,
          Plato::ScalarArray3D              & aOutput
)
{
    if(aInput.size() <= static_cast<Plato::OrdinalType>(0))
    {
        THROWERR("\nInput 2D array size is zero.\n");
    }
    if(aDomain.numCells() <= static_cast<Plato::OrdinalType>(0))
    {
        THROWERR("\nNumber of input cells, i.e. elements, is zero.\n");
    }
    if(aOutput.size() <= static_cast<Plato::OrdinalType>(0))
    {
        THROWERR("\nOutput 3D array size is zero.\n");
    }

    auto tNumCells = aDomain.numCells();
    auto tCellOrdinals = aDomain.cellOrdinals();
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        auto tCellOrdinal = tCellOrdinals[aCellOrdinal];
        for(Plato::OrdinalType tRowIndex = 0; tRowIndex < NumRowsPerCell; tRowIndex++)
        {
            for(Plato::OrdinalType tColumnIndex = 0; tColumnIndex < NumColsPerCell; tColumnIndex++)
            {
                aOutput(tCellOrdinal, tRowIndex, tColumnIndex) = aInput(aCellOrdinal, tRowIndex).dx(tColumnIndex);
            }
        }
    }, "convert AD type to Scalar type");
}
// function transform_ad_type_to_pod_3Dview
/************************************************************************//**
 *
 * \brief (3D-View) Transform automatic differentiation (AD) type to POD.
 *
 * \tparam NumRowsPerCell number of rows per cell
 * \tparam NumColsPerCell number of columns per cell
 * \tparam ADType         AD scalar type
 *
 * \param [in]     aNumCells  number of cells
 * \param [in]     aInput     AD Jacobian
 * \param [in/out] aOutput    Scalar Jacobian
 *
********************************************************************************/
template<Plato::OrdinalType NumRowsPerCell, Plato::OrdinalType NumColsPerCell, typename ADType>
inline void transform_ad_type_to_pod_3Dview(const Plato::OrdinalType& aNumCells,
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
// function transform_ad_type_to_pod_3Dview

/*************************************************************************//**
*
* \brief Assemble scalar function global value
*
* Assemble scalar function global value from local values.
*
* \fn Scalar assemble_scalar_func_value(const Plato::OrdinalType& aNumCells, const Result& aResult)
*
* \tparam Scalar typename of return value
* \tparam Result result vector view typename
*
* \param [in] aNumCells number of cells (i.e. elements)
* \param [in] aResult scalar vector
* \return global function value
*
*****************************************************************************/
template <class Scalar, class Result>
inline Scalar assemble_scalar_func_value(const Plato::OrdinalType& aNumCells, const Result& aResult)
{
  Scalar tReturnValue(0.0);
  Kokkos::parallel_reduce(Kokkos::RangePolicy<>(0, aNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType& aCellOrdinal, Scalar & aLocalValue)
  {
    aLocalValue += aResult(aCellOrdinal).val();
  }, tReturnValue);
  return tReturnValue;
}
// function assemble_scalar_func_value

/*************************************************************************//**
*
* \brief Assemble vector gradient of a scalar function
*
* \tparam NumNodesPerCell number of nodes per cells (i.e. elements)
* \tparam NumDofsPerNode number of degrees of freedom per node
* \tparam EntryOrdinal entry ordinal view type
* \tparam Gradient gradient workset view type
* \tparam ReturnVal output (i.e. assembled gradient) view type
*
* \param [in]     aNumCells      number of cells
* \param [in]     aEntryOrdinal  global indices to output vector
* \param [in]     aGradien       gradient workset - gradient values for each cell
* \param [in\out] aOutput        assembled global gradient
*
* *****************************************************************************/
template<Plato::OrdinalType NumNodesPerCell, Plato::OrdinalType NumDofsPerNode, class EntryOrdinal, class Gradient, class ReturnVal>
inline void assemble_vector_gradient(const Plato::OrdinalType& aNumCells,
                                     const EntryOrdinal& aEntryOrdinal,
                                     const Gradient& aGradient,
                                     ReturnVal& aOutput)
{
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, aNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        for(Plato::OrdinalType tNodeIndex=0; tNodeIndex < NumNodesPerCell; tNodeIndex++)
        {
            for(Plato::OrdinalType tDimIndex=0; tDimIndex < NumDofsPerNode; tDimIndex++)
            {
                Plato::OrdinalType tEntryOrdinal = aEntryOrdinal(aCellOrdinal, tNodeIndex, tDimIndex);
                Kokkos::atomic_add(&aOutput(tEntryOrdinal), aGradient(aCellOrdinal, tNodeIndex * NumDofsPerNode + tDimIndex));
            }
        }
    }, "Assemble - Vector Gradient Calculation");
}

// function assemble_vector_gradient
/*************************************************************************//**
*
* \brief Assemble vector gradient of a scalar function
*
* \tparam NumNodesPerCell number of nodes per cells (i.e. elements)
* \tparam NumDofsPerNode number of degrees of freedom per node
* \tparam EntryOrdinal entry ordinal view type
* \tparam Gradient gradient workset view type
* \tparam ReturnVal output (i.e. assembled gradient) view type
*
* \param [in]     aNumCells      number of cells
* \param [in]     aEntryOrdinal  global indices to output vector
* \param [in]     aGradien       gradient workset - gradient values for each cell
* \param [in\out] aOutput        assembled global gradient
*
* *****************************************************************************/
template<Plato::OrdinalType NumNodesPerCell, Plato::OrdinalType NumDofsPerNode, class EntryOrdinal, class Gradient, class ReturnVal>
inline void assemble_vector_gradient_fad(const Plato::OrdinalType& aNumCells,
                                         const EntryOrdinal& aEntryOrdinal,
                                         const Gradient& aGradient,
                                         ReturnVal& aOutput)
{
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, aNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        for(Plato::OrdinalType tNodeIndex=0; tNodeIndex < NumNodesPerCell; tNodeIndex++)
        {
            for(Plato::OrdinalType tDimIndex=0; tDimIndex < NumDofsPerNode; tDimIndex++)
            {
                Plato::OrdinalType tEntryOrdinal = aEntryOrdinal(aCellOrdinal, tNodeIndex, tDimIndex);
                Kokkos::atomic_add(&aOutput(tEntryOrdinal), aGradient(aCellOrdinal).dx(tNodeIndex * NumDofsPerNode + tDimIndex));
            }
        }
    }, "Assemble - Vector Gradient Calculation");
}
// function assemble_vector_gradient_fad

/*************************************************************************//**
*
* \brief Assemble vector gradient of a scalar function
*
* \tparam NumNodesPerCell number of nodes per cells (i.e. elements)
* \tparam NumDofsPerNode number of degrees of freedom per node
* \tparam EntryOrdinal entry ordinal view type
* \tparam Gradient gradient workset view type
* \tparam ReturnVal output (i.e. assembled gradient) view type
*
* \param [in]     aDomain        Plato Analyze spatial domain
* \param [in]     aEntryOrdinal  global indices to output vector
* \param [in]     aGradien       gradient workset - gradient values for each cell
* \param [in\out] aOutput        assembled global gradient
*
* *****************************************************************************/
template<Plato::OrdinalType NumNodesPerCell, Plato::OrdinalType NumDofsPerNode, class EntryOrdinal, class Gradient, class ReturnVal>
inline void
assemble_vector_gradient_fad(
    const Plato::SpatialDomain & aDomain,
    const EntryOrdinal         & aEntryOrdinal,
    const Gradient             & aGradient,
          ReturnVal            & aOutput
)
{
    auto tNumCells = aDomain.numCells();
    auto tCellOrdinals = aDomain.cellOrdinals();
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        auto tCellOrdinal = tCellOrdinals[aCellOrdinal];
        for(Plato::OrdinalType tNodeIndex=0; tNodeIndex < NumNodesPerCell; tNodeIndex++)
        {
            for(Plato::OrdinalType tDimIndex=0; tDimIndex < NumDofsPerNode; tDimIndex++)
            {
                Plato::OrdinalType tEntryOrdinal = aEntryOrdinal(tCellOrdinal, tNodeIndex, tDimIndex);
                Kokkos::atomic_add(&aOutput(tEntryOrdinal), aGradient(aCellOrdinal).dx(tNodeIndex * NumDofsPerNode + tDimIndex));
            }
        }
    }, "Assemble - Vector Gradient Calculation");
}
// function assemble_vector_gradient_fad

/*************************************************************************//**
*
* \brief Assemble scalar gradient of a scalar function
*
* \tparam NumNodesPerCell number of nodes per cell
* \tparam EntryOrdinal entry ordinal view type
* \tparam Gradient gradient workset view type
* \tparam ReturnVal output (i.e. assembled gradient) view type
*
* \param [in]     aNumCells      number of cells (i.e. elements)
* \param [in]     aEntryOrdinal  global indices to output vector
* \param [in]     aGradien       gradient workset - gradient values for each cell
* \param [in\out] aOutput        assembled global gradient
*
*****************************************************************************/
template<Plato::OrdinalType NumNodesPerCell, class EntryOrdinal, class Gradient, class ReturnVal>
inline void assemble_scalar_gradient(const Plato::OrdinalType& aNumCells,
                                     const EntryOrdinal& aEntryOrdinal,
                                     const Gradient& aGradient,
                                     ReturnVal& aOutput)
{
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, aNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
      for(Plato::OrdinalType tNodeIndex=0; tNodeIndex < NumNodesPerCell; tNodeIndex++)
      {
          Plato::OrdinalType tEntryOrdinal = aEntryOrdinal(aCellOrdinal, tNodeIndex);
          Kokkos::atomic_add(&aOutput(tEntryOrdinal), aGradient(aCellOrdinal, tNodeIndex));
      }
    }, "Assemble - Scalar Gradient Calculation");
}
// function assemble_scalar_gradient

/*************************************************************************//**
*
* \brief Assemble scalar gradient of a scalar function
*
* \tparam NumNodesPerCell number of nodes per cell
* \tparam EntryOrdinal entry ordinal view type
* \tparam Gradient gradient workset view type
* \tparam ReturnVal output (i.e. assembled gradient) view type
*
* \param [in]     aNumCells      number of cells (i.e. elements)
* \param [in]     aEntryOrdinal  global indices to output vector
* \param [in]     aGradien       gradient workset (automatic differentiation type) - gradient values for each cell
* \param [in\out] aOutput        assembled global gradient
*
*****************************************************************************/
template<Plato::OrdinalType NumNodesPerCell, class EntryOrdinal, class Gradient, class ReturnVal>
inline void assemble_scalar_gradient_fad(const Plato::OrdinalType& aNumCells,
                                         const EntryOrdinal& aEntryOrdinal,
                                         const Gradient& aGradient,
                                         ReturnVal& aOutput)
{
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, aNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
      for(Plato::OrdinalType tNodeIndex=0; tNodeIndex < NumNodesPerCell; tNodeIndex++)
      {
          Plato::OrdinalType tEntryOrdinal = aEntryOrdinal(aCellOrdinal, tNodeIndex);
          Kokkos::atomic_add(&aOutput(tEntryOrdinal), aGradient(aCellOrdinal).dx(tNodeIndex));
      }
    }, "Assemble - Scalar Gradient Calculation");
}

// function assemble_scalar_gradient_fad
/*************************************************************************//**
*
* \brief Assemble scalar gradient of a scalar function
*
* \tparam NumNodesPerCell number of nodes per cell
* \tparam EntryOrdinal entry ordinal view type
* \tparam Gradient gradient workset view type
* \tparam ReturnVal output (i.e. assembled gradient) view type
*
* \param [in]     aDomain       Plato Analyze spatial domain (block)
* \param [in]     aEntryOrdinal global indices to output vector
* \param [in]     aGradien      gradient workset (automatic differentiation type) - gradient values for each cell
* \param [in\out] aOutput       assembled global gradient
*
*****************************************************************************/
template<Plato::OrdinalType NumNodesPerCell, class EntryOrdinal, class Gradient, class ReturnVal>
inline void
assemble_scalar_gradient_fad(
    const Plato::SpatialDomain & aDomain,
    const EntryOrdinal         & aEntryOrdinal,
    const Gradient             & aGradient,
          ReturnVal            & aOutput
)
{
    auto tNumCells = aDomain.numCells();
    auto tCellOrdinals = aDomain.cellOrdinals();
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        auto tCellOrdinal = tCellOrdinals[aCellOrdinal];
        for(Plato::OrdinalType tNodeIndex=0; tNodeIndex < NumNodesPerCell; tNodeIndex++)
        {
            Plato::OrdinalType tEntryOrdinal = aEntryOrdinal(tCellOrdinal, tNodeIndex);
            Kokkos::atomic_add(&aOutput(tEntryOrdinal), aGradient(aCellOrdinal).dx(tNodeIndex));
        }
    }, "Assemble - Scalar Gradient Calculation");
}
// function assemble_scalar_gradient_fad

/***************************************************************************//**
* \brief Create control workset, i.e. set control variables for each element/cell
*        in the SpatialDomain.
*
* \tparam NumNodesPerCell     number of nodes per cell
* \tparam ControlEntryOrdinal global-to-local index map class
* \tparam Control             control variables class
* \tparam ControlWS           control workset class
*
* \param [in]     aDomain               Plato Analyze spatial domain (block)
* \param [in]     aControlEntryOrdinal  global-to-local index map
* \param [in]     aControl              1-D view of control variables
* \param [in\out] aControlWS            control variables workset
*
*******************************************************************************/
template<Plato::OrdinalType NumNodesPerCell, class ControlEntryOrdinal, class Control, class ControlWS>
inline void
workset_control_scalar_scalar(
    const Plato::SpatialDomain & aDomain,
    const ControlEntryOrdinal  & aControlEntryOrdinal,
    const Control              & aControl,
          ControlWS            & aControlWS
)
{
    auto tNumCells = aDomain.numCells();
    auto tCellOrdinals = aDomain.cellOrdinals();
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        auto tCellOrdinal = tCellOrdinals[aCellOrdinal];
        for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < NumNodesPerCell; tNodeIndex++)
        {
            Plato::OrdinalType tEntryOrdinal = aControlEntryOrdinal(tCellOrdinal, tNodeIndex);
            aControlWS(aCellOrdinal, tNodeIndex) = aControl(tEntryOrdinal);
        }
    }, "workset_control_scalar_scalar");
}
// function workset_control_scalar_scalar

/***************************************************************************//**
* \brief Create control workset, i.e. set control variables for each element/cell
*        in the parent mesh.
*
* \tparam NumNodesPerCell     number of nodes per cell
* \tparam ControlEntryOrdinal global-to-local index map class
* \tparam Control             control variables class
* \tparam ControlWS           control workset class
*
* \param [in]     aSpatialModel         Plato Analyze spatial model
* \param [in]     aControlEntryOrdinal  global-to-local index map
* \param [in]     aControl              1-D view of control variables
* \param [in\out] aControlWS            control variables workset
*
*******************************************************************************/
template<Plato::OrdinalType NumNodesPerCell, class ControlEntryOrdinal, class Control, class ControlWS>
inline void
workset_control_scalar_scalar(
          Plato::OrdinalType    aNumCells,
    const ControlEntryOrdinal & aControlEntryOrdinal,
    const Control             & aControl,
          ControlWS           & aControlWS
)
{
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, aNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < NumNodesPerCell; tNodeIndex++)
        {
            Plato::OrdinalType tEntryOrdinal = aControlEntryOrdinal(aCellOrdinal, tNodeIndex);
            aControlWS(aCellOrdinal, tNodeIndex) = aControl(tEntryOrdinal);
        }
    }, "workset_control_scalar_scalar");
}
// function workset_control_scalar_scalar

/***************************************************************************//**
* \brief Create control workset, i.e. set control variables for each element/cell
*
* \tparam NumNodesPerCell     number of nodes per cell
* \tparam ControlFad          control variables forward automatic differentation (FAD) class
* \tparam ControlEntryOrdinal global-to-local index map class
* \tparam Control             control variables class
* \tparam ControlWS           control workset FAD class
*
* \param [in]     aDomain              Plato Analyze spatial domain
* \param [in]     aControlEntryOrdinal global-to-local index map
* \param [in]     aControl             1-D view of control variables
* \param [in\out] aControlWS           control variables workset
*
*******************************************************************************/
template<Plato::OrdinalType NumNodesPerCell, class ControlFad, class ControlEntryOrdinal, class Control, class FadControlWS>
inline void
workset_control_scalar_fad(
    const Plato::SpatialDomain & aDomain,
    const ControlEntryOrdinal  & aControlEntryOrdinal,
    const Control              & aControl,
          FadControlWS         & aFadControlWS)
{
    auto tNumCells = aDomain.numCells();
    auto tCellOrdinals = aDomain.cellOrdinals();
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        auto tCellOrdinal = tCellOrdinals[aCellOrdinal];
        for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < NumNodesPerCell; tNodeIndex++)
        {
            Plato::OrdinalType tEntryOrdinal = aControlEntryOrdinal(tCellOrdinal, tNodeIndex);
            aFadControlWS(aCellOrdinal, tNodeIndex) = ControlFad( NumNodesPerCell, tNodeIndex, aControl(tEntryOrdinal));
        }
    }, "workset_control_scalar_fad");
}
// function workset_control_scalar_fad

/***************************************************************************//**
* \brief Create control workset, i.e. set control variables for each element/cell
*
* \tparam NumNodesPerCell     number of nodes per cell
* \tparam ControlFad          control variables forward automatic differentation (FAD) class
* \tparam ControlEntryOrdinal global-to-local index map class
* \tparam Control             control variables class
* \tparam ControlWS           control workset FAD class
*
* \param [in]     aNumCells             number of cells (i.e. elements)
* \param [in]     aControlEntryOrdinal  global-to-local index map
* \param [in]     aControl              1-D view of control variables
* \param [in\out] aControlWS            control variables workset
*
*******************************************************************************/
template<Plato::OrdinalType NumNodesPerCell, class ControlFad, class ControlEntryOrdinal, class Control, class FadControlWS>
inline void
workset_control_scalar_fad(
    const Plato::OrdinalType  & aNumCells,
    const ControlEntryOrdinal & aControlEntryOrdinal,
    const Control             & aControl,
          FadControlWS        & aFadControlWS)
{
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, aNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < NumNodesPerCell; tNodeIndex++)
        {
            Plato::OrdinalType tEntryOrdinal = aControlEntryOrdinal(aCellOrdinal, tNodeIndex);
            aFadControlWS(aCellOrdinal, tNodeIndex) = ControlFad( NumNodesPerCell, tNodeIndex, aControl(tEntryOrdinal));
        }
    }, "workset_control_scalar_fad");
}
// function workset_control_scalar_fad


/***************************************************************************//**
* \brief Create state workset, i.e. set state variables for each element/cell
*
* \tparam NumDofsPerCell    number of degrees of freedom per cell
* \tparam NumNodesPerCell   number of nodes per cell
* \tparam StateEntryOrdinal global-to-local index map class
* \tparam State             state variables class
* \tparam StateWS           state workset class
*
* \param [in]     aDomain             Plato Analyze spatial domain
* \param [in]     aStateEntryOrdinal  global-to-local index map
* \param [in]     aState              1-D view of state variables
* \param [in\out] aStateWS            state variables workset
*
*******************************************************************************/
template<Plato::OrdinalType NumDofsPerNode, Plato::OrdinalType NumNodesPerCell, class StateEntryOrdinal, class State, class StateWS>
inline void
workset_state_scalar_scalar(
    const Plato::SpatialDomain & aDomain,
    const StateEntryOrdinal    & aStateEntryOrdinal,
    const State                & aState,
          StateWS              & aStateWS
)
{
    auto tNumCells = aDomain.numCells();
    auto tCellOrdinals = aDomain.cellOrdinals();
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        auto tCellOrdinal = tCellOrdinals[aCellOrdinal];
        for(Plato::OrdinalType tDofIndex = 0; tDofIndex < NumDofsPerNode; tDofIndex++)
        {
            for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < NumNodesPerCell; tNodeIndex++)
            {
                Plato::OrdinalType tEntryOrdinal = aStateEntryOrdinal(tCellOrdinal, tNodeIndex, tDofIndex);
                Plato::OrdinalType tLocalDof = (tNodeIndex * NumDofsPerNode) + tDofIndex;
                aStateWS(aCellOrdinal, tLocalDof) = aState(tEntryOrdinal);
            }
        }
    }, "workset_state_scalar_scalar");
}
// function workset_state_scalar_scalar

/***************************************************************************//**
* \brief Create state workset, i.e. set state variables for each element/cell
*
* \tparam NumDofsPerCell    number of degrees of freedom per cell
* \tparam NumNodesPerCell   number of nodes per cell
* \tparam StateEntryOrdinal global-to-local index map class
* \tparam State             state variables class
* \tparam StateWS           state workset class
*
* \param [in]     aNumCells           number of cells (i.e. elements)
* \param [in]     aStateEntryOrdinal  global-to-local index map
* \param [in]     aState              1-D view of state variables
* \param [in\out] aStateWS            state variables workset
*
*******************************************************************************/
template<Plato::OrdinalType NumDofsPerNode, Plato::OrdinalType NumNodesPerCell, class StateEntryOrdinal, class State, class StateWS>
inline void
workset_state_scalar_scalar(
          Plato::OrdinalType   aNumCells,
    const StateEntryOrdinal  & aStateEntryOrdinal,
    const State              & aState,
          StateWS            & aStateWS
)
{
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, aNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        for(Plato::OrdinalType tDofIndex = 0; tDofIndex < NumDofsPerNode; tDofIndex++)
        {
            for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < NumNodesPerCell; tNodeIndex++)
            {
                Plato::OrdinalType tEntryOrdinal = aStateEntryOrdinal(aCellOrdinal, tNodeIndex, tDofIndex);
                Plato::OrdinalType tLocalDof = (tNodeIndex * NumDofsPerNode) + tDofIndex;
                aStateWS(aCellOrdinal, tLocalDof) = aState(tEntryOrdinal);
            }
        }
    }, "workset_state_scalar_scalar");
}
// function workset_state_scalar_scalar


/***************************************************************************//**
* \brief Create state workset, i.e. set state variables for each element/cell
*
* \tparam NumDofsPerCell    number of degrees of freedom per cell
* \tparam NumNodesPerCell   number of nodes per cell
* \tparam StateFad          output state forward automatic differentiation (FAD) class
* \tparam StateEntryOrdinal global-to-local index map class
* \tparam State             state variables class
* \tparam FadStateWS        state workset FAD
*
* \param [in]     aDomain            Plato Analyze spatial domain
* \param [in]     aStateEntryOrdinal global-to-local index map
* \param [in]     aState             1-D view of state variables
* \param [in\out] aFadStateWS        state variables workset
*
*******************************************************************************/
template<Plato::OrdinalType NumDofsPerNode, Plato::OrdinalType NumNodesPerCell, class StateFad, class StateEntryOrdinal, class State, class FadStateWS>
inline void
workset_state_scalar_fad(
    const Plato::SpatialDomain & aDomain,
    const StateEntryOrdinal    & aStateEntryOrdinal,
    const State                & aState,
          FadStateWS           & aFadStateWS
)
{
    auto tNumCells = aDomain.numCells();
    auto tCellOrdinals = aDomain.cellOrdinals();
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        auto tCellOrdinal = tCellOrdinals[aCellOrdinal];
        for(Plato::OrdinalType tDofIndex = 0; tDofIndex < NumDofsPerNode; tDofIndex++)
        {
            for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < NumNodesPerCell; tNodeIndex++)
            {
                Plato::OrdinalType tEntryOrdinal = aStateEntryOrdinal(tCellOrdinal, tNodeIndex, tDofIndex);
                Plato::OrdinalType tLocalDof = tNodeIndex * NumDofsPerNode + tDofIndex;
                aFadStateWS(aCellOrdinal, tLocalDof) = StateFad(NumDofsPerNode*NumNodesPerCell, tLocalDof, aState(tEntryOrdinal));
            }
        }
    }, "workset_state_scalar_fad");
}
// function workset_state_scalar_fad


/***************************************************************************//**
* \brief Create state workset, i.e. set state variables for each element/cell
*
* \tparam NumDofsPerCell    number of degrees of freedom per cell
* \tparam NumNodesPerCell   number of nodes per cell
* \tparam StateFad          output state forward automatic differentiation (FAD) class
* \tparam StateEntryOrdinal global-to-local index map class
* \tparam State             state variables class
* \tparam FadStateWS        state workset FAD
*
* \param [in]     aNumCells           number of cells (i.e. elements)
* \param [in]     aStateEntryOrdinal  global-to-local index map
* \param [in]     aState              1-D view of state variables
* \param [in\out] aFadStateWS         state variables workset
*
*******************************************************************************/
template<Plato::OrdinalType NumDofsPerNode, Plato::OrdinalType NumNodesPerCell, class StateFad, class StateEntryOrdinal, class State, class FadStateWS>
inline void
workset_state_scalar_fad(
    const Plato::OrdinalType & aNumCells,
    const StateEntryOrdinal  & aStateEntryOrdinal,
    const State              & aState,
          FadStateWS         & aFadStateWS
)
{
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, aNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        for(Plato::OrdinalType tDofIndex = 0; tDofIndex < NumDofsPerNode; tDofIndex++)
        {
            for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < NumNodesPerCell; tNodeIndex++)
            {
                Plato::OrdinalType tEntryOrdinal = aStateEntryOrdinal(aCellOrdinal, tNodeIndex, tDofIndex);
                Plato::OrdinalType tLocalDof = tNodeIndex * NumDofsPerNode + tDofIndex;
                aFadStateWS(aCellOrdinal, tLocalDof) = StateFad(NumDofsPerNode*NumNodesPerCell, tLocalDof, aState(tEntryOrdinal));
            }
        }
    }, "workset_state_scalar_fad");
}
// function workset_state_scalar_fad


/***************************************************************************//**
* \brief Create local state workset, i.e. set local state variables for each element/cell
*
* \tparam NumLocalDofsPerCell number of local degrees of freedom per cell
* \tparam State               local state variables class
* \tparam StateWS             local state workset class
*
* \param [in]     aDomain    Plato Analyze spatial domain
* \param [in]     aState     1-D view of state variables
* \param [in/out] aStateWS   state variables workset
*
*******************************************************************************/
template<Plato::OrdinalType NumLocalDofsPerCell, class State, class StateWS>
inline void
workset_local_state_scalar_scalar(
    const Plato::SpatialDomain & aDomain,
    const State                & aState,
          StateWS              & aStateWS
)
{
    auto tNumCells = aDomain.numCells();
    auto tCellOrdinals = aDomain.cellOrdinals();
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        auto tCellOrdinal = tCellOrdinals[aCellOrdinal];
        for(Plato::OrdinalType tDofIndex = 0; tDofIndex < NumLocalDofsPerCell; tDofIndex++)
        {
          Plato::OrdinalType tGlobalDof = (tCellOrdinal * NumLocalDofsPerCell) + tDofIndex;
          aStateWS(aCellOrdinal, tDofIndex) = aState(tGlobalDof);
        }
    }, "workset_local_state_scalar_scalar");
}
// function workset_local_state_scalar_scalar

/***************************************************************************//**
* \brief Create local state workset, i.e. set local state variables for each element/cell
*
* \tparam NumLocalDofsPerCell number of local degrees of freedom per cell
* \tparam State               local state variables class
* \tparam StateWS             local state workset class
*
* \param [in]     aNumCells  number of cells (i.e. elements)
* \param [in]     aState     1-D view of state variables
* \param [in/out] aStateWS   state variables workset
*
*******************************************************************************/
template<Plato::OrdinalType NumLocalDofsPerCell, class State, class StateWS>
inline void workset_local_state_scalar_scalar(const Plato::OrdinalType& aNumCells,
                                              const State& aState,
                                              StateWS& aStateWS)
{
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, aNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        for(Plato::OrdinalType tDofIndex = 0; tDofIndex < NumLocalDofsPerCell; tDofIndex++)
        {
          Plato::OrdinalType tGlobalDof = (aCellOrdinal * NumLocalDofsPerCell) + tDofIndex;
          aStateWS(aCellOrdinal, tDofIndex) = aState(tGlobalDof);
        }
    }, "workset_local_state_scalar_scalar");
}
// function workset_local_state_scalar_scalar


/***************************************************************************//**
* \brief Create local state workset, i.e. set local state variables for each element/cell
*
* \tparam NumLocalDofsPerCell number of local degrees of freedom per cell
* \tparam StateFad            output local state variables forward automatic differentiation (FAD) class
* \tparam State               local state variables class
* \tparam FadStateWS          local state workset FAD class
*
* \param [in]     aNumCells    number of cells (i.e. elements)
* \param [in]     aState       1-D view of local state variables
* \param [in/out] aFadStateWS  local state variables workset
*
*******************************************************************************/
template<Plato::OrdinalType NumLocalDofsPerCell, class StateFad, class State, class FadStateWS>
inline void workset_local_state_scalar_fad(const Plato::OrdinalType& aNumCells,
                                           const State& aState,
                                           FadStateWS& aFadStateWS)
{
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, aNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        for(Plato::OrdinalType tDofIndex = 0; tDofIndex < NumLocalDofsPerCell; tDofIndex++)
        {
          Plato::OrdinalType tGlobalDof = (aCellOrdinal * NumLocalDofsPerCell) + tDofIndex;
          aFadStateWS(aCellOrdinal,tDofIndex) = StateFad(NumLocalDofsPerCell, tDofIndex, aState(tGlobalDof));
        }
    }, "workset_local_state_scalar_fad");
}
// function workset_local_state_scalar_fad

/***************************************************************************//**
* \brief Create local state workset, i.e. set local state variables for each element/cell
*
* \tparam NumLocalDofsPerCell number of local degrees of freedom per cell
* \tparam StateFad            output local state variables forward automatic differentiation (FAD) class
* \tparam State               local state variables class
* \tparam FadStateWS          local state workset FAD class
*
* \param [in]     aDomain      Plato Analyze spatial domain
* \param [in]     aState       1-D view of local state variables
* \param [in/out] aFadStateWS  local state variables workset
*
*******************************************************************************/
template<Plato::OrdinalType NumLocalDofsPerCell, class StateFad, class State, class FadStateWS>
inline void
workset_local_state_scalar_fad(
    const Plato::SpatialDomain & aDomain,
    const State                & aState,
          FadStateWS           & aFadStateWS
)
{
    auto tNumCells = aDomain.numCells();
    auto tCellOrdinals = aDomain.cellOrdinals();
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        auto tCellOrdinal = tCellOrdinals[aCellOrdinal];
        for(Plato::OrdinalType tDofIndex = 0; tDofIndex < NumLocalDofsPerCell; tDofIndex++)
        {
          Plato::OrdinalType tGlobalDof = (tCellOrdinal * NumLocalDofsPerCell) + tDofIndex;
          aFadStateWS(aCellOrdinal,tDofIndex) = StateFad(NumLocalDofsPerCell, tDofIndex, aState(tGlobalDof));
        }
    }, "workset_local_state_scalar_fad");
}
// function workset_local_state_scalar_fad


/***************************************************************************//**
* \brief Create configuration workset, i.e. set configuration variables for each element/cell
*
* \tparam SpaceDim          number of spatial dimensions
* \tparam NumNodesPerCell   number of nodes per cell
* \tparam ConfigWS          configuration workset class
* \tparam NodeCoordinates   node coordinates container class
*
* \param [in]     aDomain          Plato Analyze spatial domain
* \param [in]     aNodeCoordinate  node coordinates
* \param [in/out] aConfigWS        configuration workset
*
*******************************************************************************/
template<Plato::OrdinalType SpaceDim, Plato::OrdinalType NumNodesPerCell, class ConfigWS, class NodeCoordinates>
inline void
workset_config_scalar(
    const Plato::SpatialDomain & aDomain,
    const NodeCoordinates      & aNodeCoordinate,
          ConfigWS             & aConfigWS
)
{
    auto tNumCells = aDomain.numCells();
    auto tCellOrdinals = aDomain.cellOrdinals();
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        auto tCellOrdinal = tCellOrdinals[aCellOrdinal];
        for(Plato::OrdinalType tDimIndex = 0; tDimIndex < SpaceDim; tDimIndex++)
        {
            for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < NumNodesPerCell; tNodeIndex++)
            {
                aConfigWS(aCellOrdinal,tNodeIndex,tDimIndex) = aNodeCoordinate(tCellOrdinal,tNodeIndex,tDimIndex);
            }
        }
    }, "workset_config_scalar");
}
// function workset_config_scalar

/***************************************************************************//**
* \brief Create configuration workset, i.e. set configuration variables for each element/cell
*
* \tparam SpaceDim          number of spatial dimensions
* \tparam NumNodesPerCell   number of nodes per cell
* \tparam ConfigWS          configuration workset class
* \tparam NodeCoordinates   node coordinates container class
*
* \param [in]     aNumCells        number of cells (i.e. elements)
* \param [in]     aNodeCoordinate  node coordinates
* \param [in/out] aConfigWS        configuration workset
*
*******************************************************************************/
template<Plato::OrdinalType SpaceDim, Plato::OrdinalType NumNodesPerCell, class ConfigWS, class NodeCoordinates>
inline void
workset_config_scalar(
          Plato::OrdinalType   aNumCells,
    const NodeCoordinates    & aNodeCoordinate,
          ConfigWS           & aConfigWS
)
{
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, aNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        for(Plato::OrdinalType tDimIndex = 0; tDimIndex < SpaceDim; tDimIndex++)
        {
            for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < NumNodesPerCell; tNodeIndex++)
            {
                aConfigWS(aCellOrdinal,tNodeIndex,tDimIndex) = aNodeCoordinate(aCellOrdinal,tNodeIndex,tDimIndex);
            }
        }
    }, "workset_config_scalar");
}
// function workset_config_scalar


/***************************************************************************//**
* \brief Create configuration workset, i.e. set configuration variables for each element/cell
*
* \tparam SpaceDim              number of spatial dimensions
* \tparam NumNodesPerCell       number of nodes per cell
* \tparam numConfigDofsPerCell  number of nodes per cell
* \tparam ConfigFad             configuration forward automatic differentiation (FAD) class
* \tparam FadConfigWS           configuration workset FAD class
* \tparam NodeCoordinates       node coordinates container class
*
* \param [in]     aDomain          Plato Analyze spatial domain
* \param [in]     aNodeCoordinate  node coordinates
* \param [in/out] aConfigWS        configuration workset
*
*******************************************************************************/
template<Plato::OrdinalType SpaceDim, Plato::OrdinalType NumNodesPerCell, Plato::OrdinalType numConfigDofsPerCell, class ConfigFad, class FadConfigWS, class NodeCoordinates>
inline void
workset_config_fad(
    const Plato::SpatialDomain & aDomain,
    const NodeCoordinates      & aNodeCoordinate,
          FadConfigWS          & aFadConfigWS
)
{
    auto tNumCells = aDomain.numCells();
    auto tCellOrdinals = aDomain.cellOrdinals();
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        auto tCellOrdinal = tCellOrdinals[aCellOrdinal];
        for(Plato::OrdinalType tDimIndex = 0; tDimIndex < SpaceDim; tDimIndex++)
        {
            for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < NumNodesPerCell; tNodeIndex++)
            {
                Plato::OrdinalType tLocalDim = tNodeIndex * SpaceDim + tDimIndex;
                aFadConfigWS(aCellOrdinal, tNodeIndex, tDimIndex) =
                        ConfigFad(numConfigDofsPerCell, tLocalDim, aNodeCoordinate(tCellOrdinal,tNodeIndex,tDimIndex));
            }
        }
    }, "workset_config_fad");
}
// function workset_config_fad

/***************************************************************************//**
* \brief Create configuration workset, i.e. set configuration variables for each element/cell
*
* \tparam SpaceDim              number of spatial dimensions
* \tparam NumNodesPerCell       number of nodes per cell
* \tparam numConfigDofsPerCell  number of nodes per cell
* \tparam ConfigFad             configuration forward automatic differentiation (FAD) class
* \tparam FadConfigWS           configuration workset FAD class
* \tparam NodeCoordinates       node coordinates container class
*
* \param [in]     aNumCells        number of cells
* \param [in]     aNodeCoordinate  node coordinates
* \param [in/out] aConfigWS        configuration workset
*
*******************************************************************************/
template<Plato::OrdinalType SpaceDim, Plato::OrdinalType NumNodesPerCell, Plato::OrdinalType numConfigDofsPerCell, class ConfigFad, class FadConfigWS, class NodeCoordinates>
inline void
workset_config_fad(
    const Plato::OrdinalType & aNumCells,
    const NodeCoordinates    & aNodeCoordinate,
          FadConfigWS        & aFadConfigWS
)
{
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, aNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        for(Plato::OrdinalType tDimIndex = 0; tDimIndex < SpaceDim; tDimIndex++)
        {
            for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < NumNodesPerCell; tNodeIndex++)
            {
                Plato::OrdinalType tLocalDim = tNodeIndex * SpaceDim + tDimIndex;
                aFadConfigWS(aCellOrdinal, tNodeIndex, tDimIndex) =
                        ConfigFad(numConfigDofsPerCell, tLocalDim, aNodeCoordinate(aCellOrdinal,tNodeIndex,tDimIndex));
            }
        }
    }, "workset_config_fad");
}
// function workset_config_fad


/***************************************************************************//**
* \brief Assemble residual vector
*
* \tparam NumNodesPerCell    number of nodes per cell
* \tparam NumDofsPerCell     number of state degree of freedom per cell
* \tparam StateEntryOrdinal  global-to-local index state map class
* \tparam Residual           input residual class
* \tparam ReturnVal          output residual class
*
* \param [in]     aNumCells           number of cells (i.e. elements)
* \param [in]     aStateEntryOrdinal  global-to-local index state map
* \param [in]     aResidual           input residual vector
* \param [in/out] aReturnValue        output residual vector
*
*******************************************************************************/
template<Plato::OrdinalType NumNodesPerCell, Plato::OrdinalType NumDofsPerNode, class StateEntryOrdinal, class Residual, class ReturnVal>
inline void
assemble_residual(
    const Plato::SpatialDomain & aDomain,
    const StateEntryOrdinal    & aStateEntryOrdinal,
    const Residual             & aResidual,
          ReturnVal            & aReturnValue)
{
    auto tNumCells = aDomain.numCells();
    auto tCellOrdinals = aDomain.cellOrdinals();
    Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0,tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        auto tCellOrdinal = tCellOrdinals[aCellOrdinal];
        for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < NumNodesPerCell; tNodeIndex++)
        {
            for(Plato::OrdinalType tDofIndex = 0; tDofIndex < NumDofsPerNode; tDofIndex++)
            {
                Plato::OrdinalType tEntryOrdinal = aStateEntryOrdinal(tCellOrdinal, tNodeIndex, tDofIndex);
                Kokkos::atomic_add(&aReturnValue(tEntryOrdinal), aResidual(aCellOrdinal,tNodeIndex*NumDofsPerNode+tDofIndex));
            }
        }
    }, "assemble_residual");
}
// function assemble_residual

/***************************************************************************//**
* \brief Assemble residual vector
*
* \tparam NumNodesPerCell    number of nodes per cell
* \tparam NumDofsPerCell     number of state degree of freedom per cell
* \tparam StateEntryOrdinal  global-to-local index state map class
* \tparam Residual           input residual class
* \tparam ReturnVal          output residual class
*
* \param [in]     aNumCells           number of cells (i.e. elements)
* \param [in]     aStateEntryOrdinal  global-to-local index state map
* \param [in]     aResidual           input residual vector
* \param [in/out] aReturnValue        output residual vector
*
*******************************************************************************/
template<Plato::OrdinalType NumNodesPerCell, Plato::OrdinalType NumDofsPerNode, class StateEntryOrdinal, class Residual, class ReturnVal>
inline void
assemble_residual(
    const Plato::OrdinalType & aNumCells,
    const StateEntryOrdinal  & aStateEntryOrdinal,
    const Residual           & aResidual,
          ReturnVal          & aReturnValue)
{
    Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0,aNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < NumNodesPerCell; tNodeIndex++)
        {
            for(Plato::OrdinalType tDofIndex = 0; tDofIndex < NumDofsPerNode; tDofIndex++)
            {
                Plato::OrdinalType tEntryOrdinal = aStateEntryOrdinal(aCellOrdinal, tNodeIndex, tDofIndex);
                Kokkos::atomic_add(&aReturnValue(tEntryOrdinal), aResidual(aCellOrdinal,tNodeIndex*NumDofsPerNode+tDofIndex));
            }
        }
    }, "assemble_residual");
}
// function assemble_residual


/***************************************************************************//**
* \brief Assemble Jacobian matrix
*
* \tparam MatrixEntriesOrdinal  matrix entries index map class
* \tparam ReturnVal             output residual class
*
* \param [in]     aNumCells            number of cells (i.e. elements)
* \param [in]     aNumRowsPerCell      number of rows
* \param [in]     aNumColumnsPerCell   number of columns
* \param [in]     aMatrixEntryOrdinal  matrix entries index map
* \param [in]     aJacobianWorkset     jacobian workset, i.e. jacobian for each element/cell
* \param [in/out] aReturnValue         output Jacobian
*
*******************************************************************************/
template<class MatrixEntriesOrdinal, class ReturnVal>
inline void assemble_jacobian(Plato::OrdinalType aNumCells,
                              Plato::OrdinalType aNumRowsPerCell,
                              Plato::OrdinalType aNumColumnsPerCell,
                              const MatrixEntriesOrdinal &aMatrixEntryOrdinal,
                              const Plato::ScalarArray3D &aJacobianWorkset,
                              ReturnVal &aReturnValue)
{
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, aNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType &aCellOrdinal)
    {
        for(Plato::OrdinalType tRowIndex = 0; tRowIndex < aNumRowsPerCell; tRowIndex++)
        {
            for(Plato::OrdinalType tColumnIndex = 0; tColumnIndex < aNumColumnsPerCell; tColumnIndex++)
            {
                Plato::OrdinalType tEntryOrdinal = aMatrixEntryOrdinal(aCellOrdinal, tRowIndex, tColumnIndex);
                Kokkos::atomic_add(&aReturnValue(tEntryOrdinal), aJacobianWorkset(aCellOrdinal,tRowIndex, tColumnIndex));
            }
        }
    }, "assemble jacobian");
}
// function assemble_jacobian


/***************************************************************************//**
* \brief Assemble Jacobian matrix
*
* \tparam MatrixEntriesOrdinal  matrix entries index map class
* \tparam ReturnVal             output residual class
*
* \param [in]     aNumCells            number of cells (i.e. elements)
* \param [in]     aNumRowsPerCell      number of rows
* \param [in]     aNumColumnsPerCell   number of columns
* \param [in]     aMatrixEntryOrdinal  matrix entries index map
* \param [in]     aJacobianWorkset     jacobian workset, i.e. jacobian for each element/cell
* \param [in/out] aReturnValue         output Jacobian
*
*******************************************************************************/
template<class MatrixEntriesOrdinal, class ReturnVal>
inline void assemble_jacobian_transpose_2(Plato::OrdinalType aNumCells,
                              Plato::OrdinalType aNumRowsPerCell,
                              Plato::OrdinalType aNumColumnsPerCell,
                              const MatrixEntriesOrdinal &aMatrixEntryOrdinal,
                              const Plato::ScalarArray3D &aJacobianWorkset,
                              ReturnVal &aReturnValue)
{
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, aNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType &aCellOrdinal)
    {
        for(Plato::OrdinalType tRowIndex = 0; tRowIndex < aNumRowsPerCell; tRowIndex++)
        {
            for(Plato::OrdinalType tColumnIndex = 0; tColumnIndex < aNumColumnsPerCell; tColumnIndex++)
            {
                Plato::OrdinalType tEntryOrdinal = aMatrixEntryOrdinal(aCellOrdinal, tColumnIndex, tRowIndex);
                Kokkos::atomic_add(&aReturnValue(tEntryOrdinal), aJacobianWorkset(aCellOrdinal,tRowIndex, tColumnIndex));
            }
        }
    }, "assemble jacobian");
}
// function assemble_jacobian


/***************************************************************************//**
* \brief Assemble Jacobian matrix
*
* \tparam MatrixEntriesOrdinal  matrix entries index map class
* \tparam Jacobian              input Jacobian workset forward automatic differentiation (FAD) class
* \tparam ReturnVal             output Jacobian FAD class
*
* \param [in]     aNumCells            number of cells (i.e. elements)
* \param [in]     aNumRowsPerCell      number of rows
* \param [in]     aNumColumnsPerCell   number of columns
* \param [in]     aMatrixEntryOrdinal  matrix entries index map
* \param [in]     aJacobianWorkset     jacobian workset, i.e. jacobian for each element/cell
* \param [in/out] aReturnValue         assembled Jacobian
*
*******************************************************************************/
template<class MatrixEntriesOrdinal, class Jacobian, class ReturnVal>
inline void
assemble_jacobian_fad(
    const Plato::SpatialDomain & aDomain,
          Plato::OrdinalType     aNumRowsPerCell,
          Plato::OrdinalType     aNumColumnsPerCell,
    const MatrixEntriesOrdinal & aMatrixEntryOrdinal,
    const Jacobian             & aJacobianWorkset,
          ReturnVal            & aReturnValue)
{
    auto tNumCells = aDomain.numCells();
    auto tCellOrdinals = aDomain.cellOrdinals();
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        auto tCellOrdinal = tCellOrdinals[aCellOrdinal];
        for(Plato::OrdinalType tRowIndex = 0; tRowIndex < aNumRowsPerCell; tRowIndex++)
        {
            for(Plato::OrdinalType tColumnIndex = 0; tColumnIndex < aNumColumnsPerCell; tColumnIndex++)
            {
                Plato::OrdinalType tEntryOrdinal = aMatrixEntryOrdinal(tCellOrdinal, tRowIndex, tColumnIndex);
                Kokkos::atomic_add(&aReturnValue(tEntryOrdinal), aJacobianWorkset(aCellOrdinal,tRowIndex).dx(tColumnIndex));
            }
        }
    }, "assemble jacobian fad");
}
// function assemble_jacobian_fad

/***************************************************************************//**
* \brief Assemble Jacobian matrix
*
* \tparam MatrixEntriesOrdinal  matrix entries index map class
* \tparam Jacobian              input Jacobian workset forward automatic differentiation (FAD) class
* \tparam ReturnVal             output Jacobian FAD class
*
* \param [in]     aDomain              Plato Analyze spatial domain
* \param [in]     aNumRowsPerCell      number of rows
* \param [in]     aNumColumnsPerCell   number of columns
* \param [in]     aMatrixEntryOrdinal  matrix entries index map
* \param [in]     aJacobianWorkset     jacobian workset, i.e. jacobian for each element/cell
* \param [in/out] aReturnValue         assembled Jacobian
*
*******************************************************************************/
template<class MatrixEntriesOrdinal, class Jacobian, class ReturnVal>
inline void
assemble_jacobian_fad(
    const Plato::OrdinalType   & aNumCells,
          Plato::OrdinalType     aNumRowsPerCell,
          Plato::OrdinalType     aNumColumnsPerCell,
    const MatrixEntriesOrdinal & aMatrixEntryOrdinal,
    const Jacobian             & aJacobianWorkset,
          ReturnVal            & aReturnValue)
{
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, aNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        for(Plato::OrdinalType tRowIndex = 0; tRowIndex < aNumRowsPerCell; tRowIndex++)
        {
            for(Plato::OrdinalType tColumnIndex = 0; tColumnIndex < aNumColumnsPerCell; tColumnIndex++)
            {
                Plato::OrdinalType tEntryOrdinal = aMatrixEntryOrdinal(aCellOrdinal, tRowIndex, tColumnIndex);
                Kokkos::atomic_add(&aReturnValue(tEntryOrdinal), aJacobianWorkset(aCellOrdinal, tRowIndex).dx(tColumnIndex));
            }
        }
    }, "assemble jacobian fad");
}
// function assemble_jacobian_fad

/***************************************************************************//**
* \brief Assemble transpose of Jacobian matrix
*
* \tparam MatrixEntriesOrdinal  matrix entries index map class
* \tparam Jacobian              input Jacobian workset forward automatic differentiation (FAD) class
* \tparam ReturnVal             output Jacobian FAD class
*
* \param [in]     aDomain              Plato Analyze spatial domain
* \param [in]     aNumRowsPerCell      number of rows
* \param [in]     aNumColumnsPerCell   number of columns
* \param [in]     aMatrixEntryOrdinal  matrix entries index map
* \param [in]     aJacobianWorkset     jacobian workset, i.e. jacobian for each element/cell
* \param [in/out] aReturnValue         assembled transpose of Jacobian
*
*******************************************************************************/
template<class MatrixEntriesOrdinal, class Jacobian, class ReturnVal>
inline void
assemble_transpose_jacobian(
    const Plato::SpatialDomain & aDomain,
          Plato::OrdinalType     aNumRowsPerCell,
          Plato::OrdinalType     aNumColumnsPerCell,
    const MatrixEntriesOrdinal & aMatrixEntryOrdinal,
    const Jacobian             & aJacobianWorkset,
          ReturnVal            & aReturnValue
)
{
    auto tNumCells = aDomain.numCells();
    auto tCellOrdinals = aDomain.cellOrdinals();
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        auto tCellOrdinal = tCellOrdinals[aCellOrdinal];
        for(Plato::OrdinalType tRowIndex = 0; tRowIndex < aNumRowsPerCell; tRowIndex++)
        {
            for(Plato::OrdinalType tColumnIndex = 0; tColumnIndex < aNumColumnsPerCell; tColumnIndex++)
            {
                Plato::OrdinalType tEntryOrdinal = aMatrixEntryOrdinal(tCellOrdinal, tColumnIndex, tRowIndex);
                Kokkos::atomic_add(&aReturnValue(tEntryOrdinal), aJacobianWorkset(aCellOrdinal, tRowIndex).dx(tColumnIndex));
            }
        }
    }, "assemble_transpose_jacobian");
}
// function assemble_transpose_jacobian

/***************************************************************************//**
* \brief Assemble transpose of Jacobian matrix
*
* \tparam MatrixEntriesOrdinal  matrix entries index map class
* \tparam Jacobian              input Jacobian workset forward automatic differentiation (FAD) class
* \tparam ReturnVal             output Jacobian FAD class
*
* \param [in]     aNumRowsPerCell      number of rows
* \param [in]     aNumColumnsPerCell   number of columns
* \param [in]     aMatrixEntryOrdinal  matrix entries index map
* \param [in]     aJacobianWorkset     jacobian workset, i.e. jacobian for each element/cell
* \param [in/out] aReturnValue         assembled transpose of Jacobian
*
*******************************************************************************/
template<class MatrixEntriesOrdinal, class Jacobian, class ReturnVal>
inline void
assemble_transpose_jacobian(
          Plato::OrdinalType     aNumCells,
          Plato::OrdinalType     aNumRowsPerCell,
          Plato::OrdinalType     aNumColumnsPerCell,
    const MatrixEntriesOrdinal & aMatrixEntryOrdinal,
    const Jacobian             & aJacobianWorkset,
          ReturnVal            & aReturnValue
)
{
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, aNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        for(Plato::OrdinalType tRowIndex = 0; tRowIndex < aNumRowsPerCell; tRowIndex++)
        {
            for(Plato::OrdinalType tColumnIndex = 0; tColumnIndex < aNumColumnsPerCell; tColumnIndex++)
            {
                Plato::OrdinalType tEntryOrdinal = aMatrixEntryOrdinal(aCellOrdinal, tColumnIndex, tRowIndex);
                Kokkos::atomic_add(&aReturnValue(tEntryOrdinal), aJacobianWorkset(aCellOrdinal, tRowIndex).dx(tColumnIndex));
            }
        }
    }, "assemble_transpose_jacobian");
}
// function assemble_transpose_jacobian

} // namespace Plato
