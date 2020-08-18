#pragma once

#include <cassert>

#include <Omega_h_mesh.hpp>

#include "ImplicitFunctors.hpp"
#include "AnalyzeMacros.hpp"
#include "Assembly.hpp"

#include "geometric/GeometricSimplexFadTypes.hpp"


namespace Plato
{

namespace Geometric
{

/******************************************************************************/
/*! Base class for workset functionality.
*/
/******************************************************************************/
template<typename SimplexGeometry>
class WorksetBase : public SimplexGeometry
{
protected:
    Plato::OrdinalType mNumCells; /*!< local number of elements */
    Plato::OrdinalType mNumNodes; /*!< local number of nodes */

    using SimplexGeometry::mNumControl;          /*!< number of control vectors, i.e. materials */
    using SimplexGeometry::mNumNodesPerCell;     /*!< number of nodes per element */

    using ControlFad    = typename Plato::Geometric::SimplexFadTypes<SimplexGeometry>::ControlFad;
    using ConfigFad     = typename Plato::Geometric::SimplexFadTypes<SimplexGeometry>::ConfigFad;

    static constexpr Plato::OrdinalType mSpaceDim = SimplexGeometry::mNumSpatialDims;
    static constexpr Plato::OrdinalType mNumConfigDofsPerCell = mSpaceDim * mNumNodesPerCell; 

    Plato::VectorEntryOrdinal<mSpaceDim,mNumControl> mControlEntryOrdinal;
    Plato::VectorEntryOrdinal<mSpaceDim,mSpaceDim>   mConfigEntryOrdinal;

    Plato::NodeCoordinate<mSpaceDim> mNodeCoordinate;

public:
    /******************************************************************************//**
     * \brief Return number of cells
     * \return number of cells
    **********************************************************************************/
    decltype(mNumCells) numCells() const
    {
        return (mNumCells);
    }

    /******************************************************************************//**
     * \brief Return number of nodes
     * \return number of nodes
    **********************************************************************************/
    decltype(mNumNodes) numNodes() const
    {
        return (mNumNodes);
    }

    /******************************************************************************//**
     * \brief Constructor
     * \param [in] aMesh mesh metadata
    **********************************************************************************/
    WorksetBase(Omega_h::Mesh& aMesh) :
            mNumCells(aMesh.nelems()),
            mNumNodes(aMesh.nverts()),
            mControlEntryOrdinal(Plato::VectorEntryOrdinal<mSpaceDim, mNumControl>(&aMesh)),
            mConfigEntryOrdinal(Plato::VectorEntryOrdinal<mSpaceDim, mSpaceDim>(&aMesh)),
            mNodeCoordinate(Plato::NodeCoordinate<mSpaceDim>(&aMesh))
    {
    }

    /******************************************************************************//**
     * \brief Get controls workset, e.g. design/optimization variables
     * \param [in] aControl controls (scalar type), as a 1-D Kokkos::View
     * \param [in/out] aFadControlWS controls workset (scalar type), as a 2-D Kokkos::View
    **********************************************************************************/
    void
    worksetControl(
        const Plato::ScalarVectorT      <Plato::Scalar> & aControl,
              Plato::ScalarMultiVectorT <Plato::Scalar> & aControlWS,
        const Plato::SpatialDomain                      & aDomain
    ) const
    {
        Plato::workset_control_scalar_scalar<mNumNodesPerCell>(
              aDomain, mControlEntryOrdinal, aControl, aControlWS);
    }

    /******************************************************************************//**
     * \brief Get controls workset, e.g. design/optimization variables
     * \param [in] aControl controls (scalar type), as a 1-D Kokkos::View
     * \param [in/out] aFadControlWS controls workset (AD type), as a 2-D Kokkos::View
    **********************************************************************************/
    void
    worksetControl(
        const Plato::ScalarVectorT      <Plato::Scalar> & aControl,
              Plato::ScalarMultiVectorT <ControlFad>    & aFadControlWS,
        const Plato::SpatialDomain                      & aDomain
    ) const
    {
        Plato::workset_control_scalar_fad<mNumNodesPerCell, ControlFad>(
              aDomain, mControlEntryOrdinal, aControl, aFadControlWS);
    }

    /******************************************************************************//**
     * \brief Get configuration workset, i.e. coordinates for each cell
     * \param [in/out] aConfigWS configuration workset (scalar type), as a 3-D Kokkos::View
    **********************************************************************************/
    void
    worksetConfig(
              Plato::ScalarArray3DT <Plato::Scalar> & aConfigWS,
        const Plato::SpatialDomain                  & aDomain
    ) const
    {
      Plato::workset_config_scalar<mSpaceDim, mNumNodesPerCell>(
              aDomain, mNodeCoordinate, aConfigWS);
    }

    /******************************************************************************//**
     * \brief Get configuration workset, i.e. coordinates for each cell
     * \param [in/out] aReturnValue configuration workset (AD type), as a 3-D Kokkos::View
    **********************************************************************************/
    void
    worksetConfig(
              Plato::ScalarArray3DT <ConfigFad> & aFadConfigWS,
        const Plato::SpatialDomain              & aDomain
    ) const
    {
      Plato::workset_config_fad<mSpaceDim, mNumNodesPerCell, mNumConfigDofsPerCell, ConfigFad>(
              aDomain, mNodeCoordinate, aFadConfigWS);
    }

#ifdef NOPE
    /******************************************************************************//**
     * \brief Assemble partial derivative with respect to configuration (X)
     *
     * \tparam WorksetType Input container, as a 2-D Kokkos::View
     * \tparam OutType Output container, as a 1-D Kokkos::View
     *
     * \param [in] aResidualWorkset residual cell workset - Scalar type
     * \param [in/out] aReturnValue assembled residual - Scalar type
    **********************************************************************************/
    template<class WorksetType, class OutType>
    void assembleVectorGradientX(const WorksetType & aWorkset, OutType & aOutput) const
    {
        Plato::assemble_vector_gradient<mNumNodesPerCell, mSpaceDim>(mNumCells, mConfigEntryOrdinal, aWorkset, aOutput);
    }

    /******************************************************************************//**
     * \brief Assemble partial derivative with respect to configuration (X) - specialized
     * for automatic differentiation types
     *
     * \tparam WorksetType Input container, as a 2-D Kokkos::View
     * \tparam OutType     Output container, as a 1-D Kokkos::View
     *
     * \param [in] aResidualWorkset residual cell workset - Scalar type
     * \param [in/out] aReturnValue assembled residual - Scalar type
    **********************************************************************************/
    template<class WorksetType, class OutType>
    void assembleVectorGradientFadX(const WorksetType & aWorkset, OutType & aOutput) const
    {
        Plato::assemble_vector_gradient_fad<mNumNodesPerCell, mSpaceDim>(mNumCells, mConfigEntryOrdinal, aWorkset, aOutput);
    }

    /******************************************************************************//**
     * \brief Assemble partial derivative with respect to controls (Z)
     *
     * \tparam WorksetType Input container, as a 2-D Kokkos::View
     * \tparam OutType Output container, as a 1-D Kokkos::View
     *
     * \param [in] aResidualWorkset residual cell workset - Scalar type
     * \param [in/out] aReturnValue assembled residual - Scalar type
    **********************************************************************************/
    template<class WorksetType, class OutType>
    void assembleScalarGradientZ(const WorksetType & aWorkset, OutType & aOutput) const
    {
        Plato::assemble_scalar_gradient<mNumNodesPerCell>(mNumCells, mControlEntryOrdinal, aWorkset, aOutput);
    }

    /******************************************************************************//**
     * \brief Assemble partial derivative with respect to controls (Z) - specialized
     * for automatic differentiation types
     *
     * \tparam WorksetType Input container, as a 2-D Kokkos::View
     * \tparam OutType     Output container, as a 1-D Kokkos::View
     *
     * \param [in] aResidualWorkset residual cell workset - Scalar type
     * \param [in/out] aReturnValue assembled residual - Scalar type
    **********************************************************************************/
    template<class WorksetType, class OutType>
    void assembleScalarGradientFadZ(const WorksetType & aWorkset, OutType & aOutput) const
    {
        Plato::assemble_scalar_gradient_fad<mNumNodesPerCell>(mNumCells, mControlEntryOrdinal, aWorkset, aOutput);
    }

    /******************************************************************************//**
     * \brief Assemble Jacobian
     *
     * \tparam MatrixEntriesOrdinalType Input container of matrix ordinal
     * \tparam JacobianWorksetType Input container, as a 2-D Kokkos::View
     * \tparam AssembledJacobianType Output container, as a 1-D Kokkos::View
     *
     * \param [in] aNumRows number of rows
     * \param [in] aNumColumns number of columns
     * \param [in] aMatrixEntryOrdinal container of Jacobian entry ordinal (local-to-global ID map)
     * \param [in] aJacobianWorkset Jacobian cell workset
     * \param [in/out] aReturnValue assembled Jacobian
    **********************************************************************************/
    template<class MatrixEntriesOrdinalType, class JacobianWorksetType, class AssembledJacobianType>
    void assembleJacobianFad(Plato::OrdinalType aNumRows,
                             Plato::OrdinalType aNumColumns,
                             const MatrixEntriesOrdinalType & aMatrixEntryOrdinal,
                             const JacobianWorksetType & aJacobianWorkset,
                             AssembledJacobianType & aReturnValue) const
    {
        Plato::assemble_jacobian_fad(mNumCells, aNumRows, aNumColumns, aMatrixEntryOrdinal, aJacobianWorkset, aReturnValue);
    }

    /******************************************************************************//**
     * \brief Assemble Jacobian
     *
     * \tparam MatrixEntriesOrdinalType Input container of matrix ordinal
     * \tparam AssembledJacobianType Output container, as a 1-D Kokkos::View
     *
     * \param [in] aNumRowsPerCell     number of rows per matrix
     * \param [in] aNumColumnsPerCell  number of columns per matrix
     * \param [in] aMatrixEntryOrdinal Jacobian entry ordinal (i.e. local-to-global ID map)
     * \param [in] aJacobianWorkset    workset of cell Jacobians
     * \param [in/out] aReturnValue    assembled transposed Jacobian
    **********************************************************************************/
    template<class MatrixEntriesOrdinalType, class AssembledJacobianType>
    void assembleJacobian(Plato::OrdinalType aNumRowsPerCell,
                          Plato::OrdinalType aNumColumnsPerCell,
                          const MatrixEntriesOrdinalType & aMatrixEntryOrdinal,
                          const Plato::ScalarArray3D & aJacobianWorkset,
                          AssembledJacobianType & aReturnValue) const
    {
        Plato::assemble_jacobian(mNumCells, aNumRowsPerCell, aNumColumnsPerCell, aMatrixEntryOrdinal, aJacobianWorkset, aReturnValue);
    }

    /******************************************************************************//**
     * \brief Assemble transpose Jacobian
     *
     * \tparam MatrixEntriesOrdinalType Input container of matrix ordinal
     * \tparam JacobianWorksetType Input container, as a 2-D Kokkos::View
     * \tparam AssembledJacobianType Output container, as a 1-D Kokkos::View
     *
     * \param [in] aNumRowsPerCell     number of rows per matrix - (use row count from untransposed matrix)
     * \param [in] aNumColumnsPerCell  number of columns per matrix - (use column count from untransposed matrix)
     * \param [in] aMatrixEntryOrdinal Jacobian entry ordinal (i.e. local-to-global ID map)
     * \param [in] aJacobianWorkset    workset of cell Jacobians
     * \param [in/out] aReturnValue    assembled transposed Jacobian
    **********************************************************************************/
    template<class MatrixEntriesOrdinalType, class JacobianWorksetType, class AssembledJacobianType>
    void assembleTransposeJacobian(Plato::OrdinalType aNumRowsPerCell,
                                   Plato::OrdinalType aNumColumnsPerCell,
                                   const MatrixEntriesOrdinalType & aMatrixEntryOrdinal,
                                   const JacobianWorksetType & aJacobianWorkset,
                                   AssembledJacobianType & aReturnValue) const
    {
        Plato::assemble_transpose_jacobian(mNumCells, aNumRowsPerCell, aNumColumnsPerCell, aMatrixEntryOrdinal, aJacobianWorkset, aReturnValue);
    }
#endif

}; // class WorksetBase

} // namespace Geometric

} // namespace Plato
