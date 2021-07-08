#ifndef HELMHOLTZ_VECTOR_FUNCTION_HPP
#define HELMHOLTZ_VECTOR_FUNCTION_HPP

#include <memory>

#include <Omega_h_mesh.hpp>
#include <Omega_h_assoc.hpp>

#include "WorksetBase.hpp"
#include "NaturalBCs.hpp"
#include "helmholtz/AbstractVectorFunction.hpp"
/* #include "helmholtz/HelmholtzSimplexFadTypes.hpp" */

namespace Plato
{

namespace Helmholtz
{

/******************************************************************************/
/*! constraint class

   This class takes as a template argument a vector function in the form:

   and manages the evaluation of the function and derivatives wrt state
   and control.
  
*/
/******************************************************************************/
template<typename PhysicsT>
class VectorFunction : public Plato::WorksetBase<PhysicsT>
{
  private:
    using Plato::WorksetBase<PhysicsT>::mNumDofsPerCell;
    using Plato::WorksetBase<PhysicsT>::mNumNodesPerCell;
    using Plato::WorksetBase<PhysicsT>::mNumDofsPerNode;
    using Plato::WorksetBase<PhysicsT>::mNumSpatialDims;
    using Plato::WorksetBase<PhysicsT>::mNumControl;
    using Plato::WorksetBase<PhysicsT>::mNumNodes;
    using Plato::WorksetBase<PhysicsT>::mNumCells;

    using Plato::WorksetBase<PhysicsT>::mGlobalStateEntryOrdinal;
    using Plato::WorksetBase<PhysicsT>::mControlEntryOrdinal;

    using Residual  = typename Plato::Evaluation<typename PhysicsT::SimplexT>::Residual;
    using Jacobian  = typename Plato::Evaluation<typename PhysicsT::SimplexT>::Jacobian;
    using GradientZ = typename Plato::Evaluation<typename PhysicsT::SimplexT>::GradientZ;

    using ResidualFunction  = std::shared_ptr<Plato::Helmholtz::AbstractVectorFunction<Residual>>;
    using JacobianFunction  = std::shared_ptr<Plato::Helmholtz::AbstractVectorFunction<Jacobian>>;
    using GradientZFunction = std::shared_ptr<Plato::Helmholtz::AbstractVectorFunction<GradientZ>>;

    static constexpr Plato::OrdinalType mNumConfigDofsPerCell = mNumSpatialDims*mNumNodesPerCell;

    std::map<std::string, ResidualFunction>  mResidualFunctions;
    std::map<std::string, JacobianFunction>  mJacobianFunctions;
    std::map<std::string, GradientZFunction> mGradientZFunctions;

    const Plato::SpatialModel & mSpatialModel;

    Plato::DataMap      & mDataMap;

  public:

    /**************************************************************************//**
    *
    * \brief Constructor
    * \param [in] aSpatialModel struct that contains the mesh, meshsets, domains, etc.
    * \param [in] aDataMap problem-specific data map
    * \param [in] aProblemParams Teuchos parameter list with input data
    *
    ******************************************************************************/
    VectorFunction(
        const Plato::SpatialModel    & aSpatialModel,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aProblemParams,
              std::string            & aProblemType
    ) :
        Plato::WorksetBase<PhysicsT>(aSpatialModel.Mesh),
        mSpatialModel  (aSpatialModel),
        mDataMap       (aDataMap)
    {
        typename PhysicsT::FunctionFactory tFunctionFactory;

        for(const auto& tDomain : mSpatialModel.Domains)
        {
          auto tName = tDomain.getDomainName();
          mResidualFunctions [tName] = tFunctionFactory.template createVectorFunction<Residual> (tDomain, aDataMap, aProblemParams, aProblemType);
          mJacobianFunctions [tName] = tFunctionFactory.template createVectorFunction<Jacobian> (tDomain, aDataMap, aProblemParams, aProblemType);
          mGradientZFunctions[tName] = tFunctionFactory.template createVectorFunction<GradientZ>(tDomain, aDataMap, aProblemParams, aProblemType);
        }
    }

    /**************************************************************************//**
    * \brief Constructor
    * \param [in] aSpatialModel struct that contains the mesh, meshsets, domains, etc.
    * \param [in] aDataMap problem-specific data map
    ******************************************************************************/
    VectorFunction(
        const Plato::SpatialModel & aSpatialModel,
              Plato::DataMap      & aDataMap
    ) :
        Plato::WorksetBase<PhysicsT>(aSpatialModel.Mesh),
        mSpatialModel  (aSpatialModel),
        mDataMap       (aDataMap)
    {
    }

    /**************************************************************************//**
    * \brief Return number of nodes on the mesh
    * \return number of nodes
    ******************************************************************************/
    Plato::OrdinalType numNodes() const
    {
        return (mNumNodes);
    }

    /**************************************************************************//**
    * \brief Return number of elements/cells on the mesh
    * \return number of elements
    ******************************************************************************/
    Plato::OrdinalType numCells() const
    {
        return (mNumCells);
    }

    /**************************************************************************//**
    * \brief Return total number of global degrees of freedom
    * \return total number of global degrees of freedom
    ******************************************************************************/
    Plato::OrdinalType numDofsPerCell() const
    {
        return (mNumDofsPerCell);
    }

    /**************************************************************************//**
    * \brief Return total number of nodes per cell/element
    * \return total number of nodes per cell/element
    ******************************************************************************/
    Plato::OrdinalType numNodesPerCell() const
    {
        return (mNumNodesPerCell);
    }

    /**************************************************************************//**
    * \brief Return number of degrees of freedom per node
    * \return number of degrees of freedom per node
    ******************************************************************************/
    Plato::OrdinalType numDofsPerNode() const
    {
        return (mNumDofsPerNode);
    }

    /**************************************************************************//**
    * \brief Return number of control vectors/fields, e.g. number of materials.
    * \return number of control vectors
    ******************************************************************************/
    Plato::OrdinalType numControlsPerNode() const
    {
        return (mNumControl);
    }

    /**************************************************************************//**
    *
    * \brief Allocate residual evaluator
    * \param [in] aResidual residual evaluator
    * \param [in] aJacobian Jacobian evaluator
    * \param [in] aName Name of the mesh domain
    *
    ******************************************************************************/
    void
    setEvaluator(
        const ResidualFunction & aResidual,
        const JacobianFunction & aJacobian,
              std::string        aName
    )
    {
        mResidualFunctions[aName] = aResidual;
        mJacobianFunctions[aName] = aJacobian;
    }

    /**************************************************************************//**
    *
    * \brief Return local number of degrees of freedom
    *
    ******************************************************************************/
    Plato::OrdinalType size() const
    {
      return mNumNodes*mNumDofsPerNode;
    }

    /****************************************************************************//**
    * \brief Pure virtual function to get output solution data
    * \param [in] state solution database
    * \return output state solution database
    ********************************************************************************/
    Plato::Solutions getSolutionStateOutputData(const Plato::Solutions &aSolutions) const
    {
        auto tFirstBlockName = mSpatialModel.Domains.front().getDomainName();
        auto tItr = mResidualFunctions.find(tFirstBlockName);
        if(tItr == mResidualFunctions.end())
            { THROWERR(std::string("Element block with name '") + tFirstBlockName + "is not defined in residual function to element block map.") }
        return tItr->second->getSolutionStateOutputData(aSolutions);
    }

    /**************************************************************************/
    Plato::ScalarVector
    value(
        const Plato::ScalarVector & aState,
        const Plato::ScalarVector & aControl,
              Plato::Scalar         aTimeStep = 0.0
    /**************************************************************************/
    ) const
    {
        using ConfigScalar  = typename Residual::ConfigScalarType;
        using StateScalar   = typename Residual::StateScalarType;
        using ControlScalar = typename Residual::ControlScalarType;
        using ResultScalar  = typename Residual::ResultScalarType;


        Plato::ScalarVector  tReturnValue("Assembled Residual",mNumDofsPerNode*mNumNodes);

        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            auto tName     = tDomain.getDomainName();

            // Workset state
            //
            Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset", tNumCells, mNumDofsPerCell);
            Plato::WorksetBase<PhysicsT>::worksetState(aState, tStateWS, tDomain);

            // Workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", tNumCells, mNumNodesPerCell);
            Plato::WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS, tDomain);

            // Workset config
            //
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset", tNumCells, mNumNodesPerCell, mNumSpatialDims);
            Plato::WorksetBase<PhysicsT>::worksetConfig(tConfigWS, tDomain);

            // create result
            //
            Plato::ScalarMultiVectorT<ResultScalar> tResidual("Cells Residual", tNumCells, mNumDofsPerCell);

            // evaluate function
            //
            mResidualFunctions.at(tName)->evaluate( tStateWS, tControlWS, tConfigWS, tResidual, aTimeStep );

            // assemble to return view
            //
            Plato::WorksetBase<PhysicsT>::assembleResidual( tResidual, tReturnValue, tDomain );

        }

        return tReturnValue;
    }

    /**************************************************************************/
    Teuchos::RCP<Plato::CrsMatrixType>
    gradient_u(
        const Plato::ScalarVector & aState,
        const Plato::ScalarVector & aControl,
              Plato::Scalar         aTimeStep = 0.0
    ) const
    /**************************************************************************/
    {
        using ConfigScalar  = typename Jacobian::ConfigScalarType;
        using StateScalar   = typename Jacobian::StateScalarType;
        using ControlScalar = typename Jacobian::ControlScalarType;
        using ResultScalar  = typename Jacobian::ResultScalarType;

        // create return matrix
        //
        auto tMesh = mSpatialModel.Mesh;
        Teuchos::RCP<Plato::CrsMatrixType> tJacobianMat =
                Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumDofsPerNode, mNumDofsPerNode>( &tMesh );

        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            auto tName     = tDomain.getDomainName();

            // Workset config
            //
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset", tNumCells, mNumNodesPerCell, mNumSpatialDims);
            Plato::WorksetBase<PhysicsT>::worksetConfig(tConfigWS, tDomain);

            // Workset state
            //
            Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset", tNumCells, mNumDofsPerCell);
            Plato::WorksetBase<PhysicsT>::worksetState(aState, tStateWS, tDomain);

            // Workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", tNumCells, mNumNodesPerCell);
            Plato::WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS, tDomain);

            // create return view
            //
            Plato::ScalarMultiVectorT<ResultScalar> tJacobian("JacobianState", tNumCells, mNumDofsPerCell);

            // evaluate function
            //
            mJacobianFunctions.at(tName)->evaluate( tStateWS, tControlWS, tConfigWS, tJacobian, aTimeStep );

            // assembly to return matrix
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumDofsPerNode, mNumDofsPerNode>
                tJacobianMatEntryOrdinal( tJacobianMat, &tMesh );

            auto tJacobianMatEntries = tJacobianMat->entries();
            Plato::WorksetBase<PhysicsT>::assembleJacobianFad
                (mNumDofsPerCell, mNumDofsPerCell, tJacobianMatEntryOrdinal, tJacobian, tJacobianMatEntries, tDomain);
        }

        return tJacobianMat;
    }

    /**************************************************************************/
    Teuchos::RCP<Plato::CrsMatrixType>
    gradient_z(
        const Plato::ScalarVector & aState,
        const Plato::ScalarVector & aControl,
              Plato::Scalar         aTimeStep = 0.0
    /**************************************************************************/
    ) const
    {
        using ConfigScalar  = typename GradientZ::ConfigScalarType;
        using StateScalar   = typename GradientZ::StateScalarType;
        using ControlScalar = typename GradientZ::ControlScalarType;
        using ResultScalar  = typename GradientZ::ResultScalarType;

        // create return matrix
        //
        auto tMesh = mSpatialModel.Mesh;
        Teuchos::RCP<Plato::CrsMatrixType> tJacobianMat =
                Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumControl, mNumDofsPerNode>( &tMesh );

        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            auto tName     = tDomain.getDomainName();

            // Workset config
            //
            Plato::ScalarArray3DT<ConfigScalar>
                tConfigWS("Config Workset", tNumCells, mNumNodesPerCell, mNumSpatialDims);
            Plato::WorksetBase<PhysicsT>::worksetConfig(tConfigWS, tDomain);

            // Workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", tNumCells, mNumNodesPerCell);
            Plato::WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS, tDomain);
 
            // Workset state
            //
            Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset", tNumCells, mNumDofsPerCell);
            Plato::WorksetBase<PhysicsT>::worksetState(aState, tStateWS, tDomain);

            // create result 
            //
            Plato::ScalarMultiVectorT<ResultScalar> tJacobian("JacobianControl", tNumCells, mNumDofsPerCell);

            // evaluate function 
            //
            mGradientZFunctions.at(tName)->evaluate( tStateWS, tControlWS, tConfigWS, tJacobian, aTimeStep );

            // assembly to return matrix
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumControl, mNumDofsPerNode>
              tJacobianMatEntryOrdinal( tJacobianMat, &tMesh );

            auto tJacobianMatEntries = tJacobianMat->entries();
            Plato::WorksetBase<PhysicsT>::assembleTransposeJacobian
                (mNumDofsPerCell, mNumNodesPerCell, tJacobianMatEntryOrdinal, tJacobian, tJacobianMatEntries, tDomain);
        }

        return tJacobianMat;
    }

};
// class VectorFunction

} // namespace Helmholtz

} // namespace Plato

#endif
