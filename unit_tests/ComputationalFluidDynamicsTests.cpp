/*
 * ComputationalFluidDynamicsTests.cpp
 *
 *  Created on: Oct 13, 2020
 */

#include <unordered_map>

#include "Simplex.hpp"
#include "WorksetBase.hpp"
#include "SpatialModel.hpp"
#include "SimplexFadTypes.hpp"
#include "PlatoAbstractProblem.hpp"

namespace Plato
{

/******************************************************************************//**
 * \fn tolower
 * \brief Convert uppercase word to lowercase.
 * \param [in] aInput word
 * \return lowercase word
**********************************************************************************/
inline std::string tolower(const std::string& aInput)
{
    std::locale tLocale;
    std::ostringstream tOutput;
    for (auto& tChar : aInput)
    {
        tOutput << std::tolower(tChar,tLocale);
    }
    return (tOutput.str());
}
// function tolower

/***************************************************************************//**
 *  \brief Base class for simplex-based fluid mechanics problems
 *  \tparam SpaceDim    (integer) spatial dimensions
 *  \tparam NumControls (integer) number of design variable fields (default = 1)
 ******************************************************************************/
template<Plato::OrdinalType SpaceDim, Plato::OrdinalType NumControls = 1>
class SimplexFluidMechanics: public Plato::Simplex<SpaceDim>
{
public:
    using Plato::Simplex<SpaceDim>::mNumSpatialDims;  /*!< number of spatial dimensions */
    using Plato::Simplex<SpaceDim>::mNumNodesPerCell; /*!< number of nodes per simplex cell */

    static constexpr Plato::OrdinalType mNumControls = NumControls; /*!< number of design variable fields */

    static constexpr Plato::OrdinalType mNumVoigtTerms =                      /*!< number of fluid stress terms */
        (SpaceDim == 3) ? 6 : ((SpaceDim == 2) ? 3 : (((SpaceDim == 1) ? 1 : 0)));

    static constexpr Plato::OrdinalType mNumMassDofsPerNode     = 1; /*!< number of continuity degrees of freedom per node */
    static constexpr Plato::OrdinalType mNumMassDofsPerCell     = mNumMassDofsPerNode * mNumNodesPerCell; /*!< number of continuity degrees of freedom per cell */
    static constexpr Plato::OrdinalType mNumEnergyDofsPerNode   = 1; /*!< number energy degrees of freedom per node */
    static constexpr Plato::OrdinalType mNumEnergyDofsPerCell   = mNumEnergyDofsPerNode * mNumNodesPerCell; /*!< number of energy degrees of freedom per cell */
    static constexpr Plato::OrdinalType mNumMomentumDofsPerNode = SpaceDim; /*!< number of momentum degrees of freedom per node */
    static constexpr Plato::OrdinalType mNumMomentumDofsPerCell = mNumMomentumDofsPerNode * mNumNodesPerCell; /*!< number of momentum degrees of freedom per cell */
};
// class SimplexFluidDynamics

namespace Hyperbolic
{

namespace FluidMechanics
{

template<typename SimplexPhysics>
struct SimplexFadTypes
{
    using MassFad = Sacado::Fad::SFad<Plato::Scalar, SimplexPhysics::mNumMassDofsPerCell>;
    using ControlFad = Sacado::Fad::SFad<Plato::Scalar, SimplexPhysics::mNumNodesPerCell>;
    using ConfigFad = Sacado::Fad::SFad<Plato::Scalar, SimplexPhysics::mNumSpatialDims *  SimplexPhysics::mNumNodesPerCell>;
    using EnergyFad = Sacado::Fad::SFad<Plato::Scalar, SimplexPhysics::mNumEnergyDofsPerNode * SimplexPhysics::mNumNodesPerCell>;
    using MomentumFad = Sacado::Fad::SFad<Plato::Scalar, SimplexPhysics::mNumMomentumDofsPerNode * SimplexPhysics::mNumNodesPerCell>;
};

struct State
{
private:
    Plato::Scalar mTimeStep = 1.0;
    Plato::Scalar mCurrentTime = 0.0;
    std::unordered_map<std::string, Plato::ScalarVector> mStates;

public:
    Plato::Scalar time() const
    {
        return mCurrentTime;
    }
    void time(Plato::Scalar aInput)
    {
        mCurrentTime = aInput;
    }
    Plato::Scalar timeStep() const
    {
        return mTimeStep;
    }
    void timeStep(Plato::Scalar aInput)
    {
        mTimeStep = aInput;
    }
    Plato::ScalarVector get(const std::string& aTag)
    {
        auto tLowerTag = Plato::tolower(aTag);
        auto tItr = mStates.find(tLowerTag);
        if(tItr == mStates.end())
        {
            THROWERR(std::string("State with tag '") + aTag + "' is not defined in state map.")
        }
        return tItr->second;
    }
    void set(const std::string& aTag, const Plato::ScalarVector& aInput)
    {
        auto tLowerTag = Plato::tolower(aTag);
        mStates[tLowerTag] = aInput;
    }
};

/***************************************************************************//**
 *  \brief Base class for automatic differentiation types used in fluid problems
 *  \tparam SpaceDim    (integer) spatial dimensions
 *  \tparam SimplexPhysicsT simplex fluid dynamic physics type
 ******************************************************************************/
template <typename SimplexPhysicsT>
struct EvaluationTypes
{
    static constexpr Plato::OrdinalType mNumControls = SimplexPhysicsT::mNumControls; /*!< number of design variable fields */
    static constexpr Plato::OrdinalType mNumSpatialDims = SimplexPhysicsT::mNumSpatialDims; /*!< number of spatial dimensions */
    static constexpr Plato::OrdinalType mNumNodesPerCell = SimplexPhysicsT::mNumNodesPerCell; /*!< number of nodes per simplex cell */
};

template <typename SimplexPhysicsT>
struct ResidualTypes : EvaluationTypes<SimplexPhysicsT>
{
    using ControlScalarType           = Plato::Scalar;
    using ConfigScalarType            = Plato::Scalar;
    using ResultScalarType            = Plato::Scalar;
    using CurrentMassScalarType       = Plato::Scalar;
    using CurrentEnergyScalarType     = Plato::Scalar;
    using CurrentMomentumScalarType   = Plato::Scalar;
    using PreviousMassScalarType      = Plato::Scalar;
    using PreviousEnergyScalarType    = Plato::Scalar;
    using PreviousMomentumScalarType  = Plato::Scalar;
    using MomentumPredictorScalarType = Plato::Scalar;
};

template <typename SimplexPhysicsT>
struct GradCurrentMomentumTypes : EvaluationTypes<SimplexPhysicsT>
{
  using FadType = typename Plato::Hyperbolic::FluidMechanics::SimplexFadTypes<SimplexPhysicsT>::MomentumFad;

  using ControlScalarType           = Plato::Scalar;
  using ConfigScalarType            = Plato::Scalar;
  using ResultScalarType            = FadType;
  using CurrentMassScalarType       = Plato::Scalar;
  using CurrentEnergyScalarType     = Plato::Scalar;
  using CurrentMomentumScalarType   = FadType;
  using PreviousMassScalarType      = Plato::Scalar;
  using PreviousEnergyScalarType    = Plato::Scalar;
  using PreviousMomentumScalarType  = Plato::Scalar;
  using MomentumPredictorScalarType = Plato::Scalar;
};

template <typename SimplexPhysicsT>
struct GradCurrentEnergyTypes : EvaluationTypes<SimplexPhysicsT>
{
  using FadType = typename Plato::Hyperbolic::FluidMechanics::SimplexFadTypes<SimplexPhysicsT>::EnergyFad;

  using ControlScalarType           = Plato::Scalar;
  using ConfigScalarType            = Plato::Scalar;
  using ResultScalarType            = FadType;
  using CurrentMassScalarType       = Plato::Scalar;
  using CurrentEnergyScalarType     = FadType;
  using CurrentMomentumScalarType   = Plato::Scalar;
  using PreviousMassScalarType      = Plato::Scalar;
  using PreviousEnergyScalarType    = Plato::Scalar;
  using PreviousMomentumScalarType  = Plato::Scalar;
  using MomentumPredictorScalarType = Plato::Scalar;
};

template <typename SimplexPhysicsT>
struct GradCurrentMassTypes : EvaluationTypes<SimplexPhysicsT>
{
  using FadType = typename Plato::Hyperbolic::FluidMechanics::SimplexFadTypes<SimplexPhysicsT>::MassFad;

  using ControlScalarType           = Plato::Scalar;
  using ConfigScalarType            = Plato::Scalar;
  using ResultScalarType            = FadType;
  using CurrentMassScalarType       = FadType;
  using CurrentEnergyScalarType     = Plato::Scalar;
  using CurrentMomentumScalarType   = Plato::Scalar;
  using PreviousMassScalarType      = Plato::Scalar;
  using PreviousEnergyScalarType    = Plato::Scalar;
  using PreviousMomentumScalarType  = Plato::Scalar;
  using MomentumPredictorScalarType = Plato::Scalar;
};

template <typename SimplexPhysicsT>
struct GradPreviousMomentumTypes : EvaluationTypes<SimplexPhysicsT>
{
  using FadType = typename Plato::Hyperbolic::FluidMechanics::SimplexFadTypes<SimplexPhysicsT>::MomentumFad;

  using ControlScalarType           = Plato::Scalar;
  using ConfigScalarType            = Plato::Scalar;
  using ResultScalarType            = FadType;
  using CurrentMassScalarType       = Plato::Scalar;
  using CurrentEnergyScalarType     = Plato::Scalar;
  using CurrentMomentumScalarType   = Plato::Scalar;
  using PreviousMassScalarType      = Plato::Scalar;
  using PreviousEnergyScalarType    = Plato::Scalar;
  using PreviousMomentumScalarType  = FadType;
  using MomentumPredictorScalarType = Plato::Scalar;
};

template <typename SimplexPhysicsT>
struct GradPreviousEnergyTypes : EvaluationTypes<SimplexPhysicsT>
{
  using FadType = typename Plato::Hyperbolic::FluidMechanics::SimplexFadTypes<SimplexPhysicsT>::EnergyFad;

  using ControlScalarType           = Plato::Scalar;
  using ConfigScalarType            = Plato::Scalar;
  using ResultScalarType            = FadType;
  using CurrentMassScalarType       = Plato::Scalar;
  using CurrentEnergyScalarType     = Plato::Scalar;
  using CurrentMomentumScalarType   = Plato::Scalar;
  using PreviousMassScalarType      = Plato::Scalar;
  using PreviousEnergyScalarType    = FadType;
  using PreviousMomentumScalarType  = Plato::Scalar;
  using MomentumPredictorScalarType = Plato::Scalar;
};

template <typename SimplexPhysicsT>
struct GradPreviousMassTypes : EvaluationTypes<SimplexPhysicsT>
{
  using FadType = typename Plato::Hyperbolic::FluidMechanics::SimplexFadTypes<SimplexPhysicsT>::MassFad;

  using ControlScalarType           = Plato::Scalar;
  using ConfigScalarType            = Plato::Scalar;
  using ResultScalarType            = FadType;
  using CurrentMassScalarType       = Plato::Scalar;
  using CurrentEnergyScalarType     = Plato::Scalar;
  using CurrentMomentumScalarType   = Plato::Scalar;
  using PreviousMassScalarType      = FadType;
  using PreviousEnergyScalarType    = Plato::Scalar;
  using PreviousMomentumScalarType  = Plato::Scalar;
  using MomentumPredictorScalarType = Plato::Scalar;
};

template <typename SimplexPhysicsT>
struct GradMomentumPredictorTypes : EvaluationTypes<SimplexPhysicsT>
{
  using FadType = typename Plato::Hyperbolic::FluidMechanics::SimplexFadTypes<SimplexPhysicsT>::MomentumFad;

  using ControlScalarType           = Plato::Scalar;
  using ConfigScalarType            = Plato::Scalar;
  using ResultScalarType            = FadType;
  using CurrentMassScalarType       = Plato::Scalar;
  using CurrentEnergyScalarType     = Plato::Scalar;
  using CurrentMomentumScalarType   = Plato::Scalar;
  using PreviousMassScalarType      = Plato::Scalar;
  using PreviousEnergyScalarType    = Plato::Scalar;
  using PreviousMomentumScalarType  = Plato::Scalar;
  using MomentumPredictorScalarType = FadType;
};

template <typename SimplexPhysicsT>
struct GradConfigTypes : EvaluationTypes<SimplexPhysicsT>
{
  using FadType = typename SimplexFadTypes<SimplexPhysicsT>::ConfigFad;

  using ControlScalarType           = Plato::Scalar;
  using ConfigScalarType            = FadType;
  using ResultScalarType            = FadType;
  using CurrentMassScalarType       = Plato::Scalar;
  using CurrentEnergyScalarType     = Plato::Scalar;
  using CurrentMomentumScalarType   = Plato::Scalar;
  using PreviousMassScalarType      = Plato::Scalar;
  using PreviousEnergyScalarType    = Plato::Scalar;
  using PreviousMomentumScalarType  = Plato::Scalar;
  using MomentumPredictorScalarType = Plato::Scalar;
};

template <typename SimplexPhysicsT>
struct GradControlTypes : EvaluationTypes<SimplexPhysicsT>
{
  using FadType = typename SimplexFadTypes<SimplexPhysicsT>::ControlFad;

  using ControlScalarType          = FadType;
  using ConfigScalarType           = Plato::Scalar;
  using ResultScalarType           = FadType;
  using CurrentMassScalarType      = Plato::Scalar;
  using CurrentEnergyScalarType    = Plato::Scalar;
  using CurrentMomentumScalarType  = Plato::Scalar;
  using PreviousMassScalarType     = Plato::Scalar;
  using PreviousEnergyScalarType   = Plato::Scalar;
  using PreviousMomentumScalarType = Plato::Scalar;
};

template <typename SimplexPhysicsT>
struct Evaluation
{
   using Residual         = ResidualTypes<SimplexPhysicsT>;
   using GradConfig       = GradControlTypes<SimplexPhysicsT>;
   using GradControl      = GradConfigTypes<SimplexPhysicsT>;
   using GradMassCurr     = GradCurrentMassTypes<SimplexPhysicsT>;
   using GradMassPrev     = GradCurrentMassTypes<SimplexPhysicsT>;
   using GradEnergyCurr   = GradCurrentEnergyTypes<SimplexPhysicsT>;
   using GradEnergyPrev   = GradPreviousEnergyTypes<SimplexPhysicsT>;
   using GradMomentumCurr = GradPreviousMomentumTypes<SimplexPhysicsT>;
   using GradMomentumPrev = GradPreviousMomentumTypes<SimplexPhysicsT>;
   using GradMomentumPred = GradMomentumPredictorTypes<SimplexPhysicsT>;
};

}
// namespace FluidMechanics

namespace Momentum
{

/******************************************************************************/
/*! constraint class

   This class takes as a template argument a vector function in the form:

   F = F(\phi, U^k, V^k, A^k, X)

   and manages the evaluation of the function and derivatives with respect to
   state, U^k, state dot, V^k, state dot dot, V^k, and control, X.

*/
/******************************************************************************/
template<typename PhysicsT>
class VectorFunction
{
private:
    static constexpr auto mNumControls          = PhysicsT::SimplexT::mNumControl;             /*!< number of design variable fields */
    static constexpr auto mNumSpatialDims       = PhysicsT::SimplexT::mNumSpatialDims;         /*!< number of spatial dimensions */
    static constexpr auto mNumNodesPerCell      = PhysicsT::SimplexT::mNumNodesPerCell;        /*!< number of nodes per cell */
    static constexpr auto mNumVelDofsPerCell    = PhysicsT::SimplexT::mNumMomentumDofsPerCell; /*!< number of momentum dofs per cell */
    static constexpr auto mNumTempDofsPerCell   = PhysicsT::SimplexT::mNumEnergyDofsPerCell;   /*!< number of energy dofs per cell */
    static constexpr auto mNumPressDofsPerCell  = PhysicsT::SimplexT::mNumMassDofsPerCell;     /*!< number of mass dofs per cell */
    static constexpr auto mNumVelDofsPerNode   = PhysicsT::SimplexT::mNumMomentumDofsPerNode; /*!< number of momentum dofs per node */
    static constexpr auto mNumTempDofsPerNode   = PhysicsT::SimplexT::mNumEnergyDofsPerNode;   /*!< number of energy dofs per node */
    static constexpr auto mNumPressDofsPerNode  = PhysicsT::SimplexT::mNumMassDofsPerNode;     /*!< number of mass dofs per node */
    static constexpr auto mNumConfigDofsPerCell = mNumSpatialDims * mNumNodesPerCell;          /*!< number of configuration degrees of freedom per cell */

    using Residual         = typename Plato::Hyperbolic::FluidMechanics::Evaluation<typename PhysicsT::SimplexT>::Residual;
    using GradConfig       = typename Plato::Hyperbolic::FluidMechanics::Evaluation<typename PhysicsT::SimplexT>::GradConfig;
    using GradControl      = typename Plato::Hyperbolic::FluidMechanics::Evaluation<typename PhysicsT::SimplexT>::GradControl;
    using GradMassCurr     = typename Plato::Hyperbolic::FluidMechanics::Evaluation<typename PhysicsT::SimplexT>::GradMassCurr;
    using GradMassPrev     = typename Plato::Hyperbolic::FluidMechanics::Evaluation<typename PhysicsT::SimplexT>::GradMassPrev;
    using GradEnergyCurr   = typename Plato::Hyperbolic::FluidMechanics::Evaluation<typename PhysicsT::SimplexT>::GradEnergyCurr;
    using GradEnergyPrev   = typename Plato::Hyperbolic::FluidMechanics::Evaluation<typename PhysicsT::SimplexT>::GradEnergyPrev;
    using GradMomentumCurr = typename Plato::Hyperbolic::FluidMechanics::Evaluation<typename PhysicsT::SimplexT>::GradMomentumCurr;
    using GradMomentumPrev = typename Plato::Hyperbolic::FluidMechanics::Evaluation<typename PhysicsT::SimplexT>::GradMomentumPrev;
    using GradMomentumPred = typename Plato::Hyperbolic::FluidMechanics::Evaluation<typename PhysicsT::SimplexT>::GradMomentumPred;

    Plato::DataMap& mDataMap;
    const Plato::SpatialModel& mSpatialModel;

    Plato::NodeCoordinate<mNumSpatialDims> mNodeCoordinate; /*!< node coordinates metadata */

    Plato::VectorEntryOrdinal<mNumSpatialDims, mNumControls>         mControlEntryOrdinal;    /*!< control local-to-global ID map */
    Plato::VectorEntryOrdinal<mNumSpatialDims, mNumSpatialDims>      mConfigEntryOrdinal;     /*!< configuration local-to-global ID map */
    Plato::VectorEntryOrdinal<mNumSpatialDims, mNumVelDofsPerCell>   mVelStateEntryOrdinal;   /*!< momentum state local-to-global ID map */
    Plato::VectorEntryOrdinal<mNumSpatialDims, mNumTempDofsPerCell>  mTempStateEntryOrdinal;  /*!< energy state local-to-global ID map */
    Plato::VectorEntryOrdinal<mNumSpatialDims, mNumPressDofsPerCell> mPressStateEntryOrdinal; /*!< mass state local-to-global ID map */

public:
    /**************************************************************************//**
    * \brief Constructor
    * \param [in] aSpatialModel struct that contains the mesh, meshsets, domains, etc.
    * \param [in] aDataMap      problem-specific data map
    * \param [in] aParamList    Teuchos parameter list with input data
    * \param [in] aProblemType  problem type
    ******************************************************************************/
    VectorFunction(const Plato::SpatialModel &aSpatialModel,
                   Plato::DataMap &aDataMap,
                   Teuchos::ParameterList &aParamList,
                   std::string &aProblemType) :
        Plato::WorksetBase<PhysicsT>(aSpatialModel.Mesh),
        mSpatialModel(aSpatialModel),
        mDataMap(aDataMap)
    {
    }

    /**************************************************************************//**
    * \brief Return total number of momentum degrees of freedom
    ******************************************************************************/
    Plato::OrdinalType size() const
    {
        auto tNumNodes = mSpatialModel.Mesh.nverts();
        return (tNumNodes * mNumVelDofsPerCell);
    }

    Plato::ScalarVector
    value(Plato::Hyperbolic::FluidMechanics::State& aState) const
    {
        using ControlScalarT   = typename Residual::ControlScalarType;
        using ConfigScalarT    = typename Residual::ConfigScalarType;
        using ResultScalarT    = typename Residual::ResultScalarType;
        using VelPredScalarT   = typename Residual::MomentumPredictorScalarType;
        using PrevVelScalarT   = typename Residual::PreviousMomentumScalarType;
        using PrevTempScalarT  = typename Residual::PreviousEnergyScalarType;
        using PrevPressScalarT = typename Residual::PreviousMassScalarType;

        auto tLength = this->size();
        Plato::ScalarVector tReturnValue("Assembled Residual", tLength);

        // internal force contribution
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            auto tName     = tDomain.getDomainName();

            Plato::ScalarMultiVectorT<PrevVelScalarT> tVelPredictorWS("Velocity Predictor Workset", tNumCells, mNumVelDofsPerCell);
            Plato::workset_state_scalar_scalar<mNumVelDofsPerNode, mNumNodesPerCell>
                (tDomain, mVelStateEntryOrdinal, aState.get("velocity predictor"), tVelPredictorWS);

            Plato::ScalarMultiVectorT<PrevVelScalarT> tPrevMomentumWS("Previous Velocity Workset", tNumCells, mNumVelDofsPerCell);
            Plato::workset_state_scalar_scalar<mNumVelDofsPerNode, mNumNodesPerCell>
                (tDomain, mVelStateEntryOrdinal, aState.get("previous velocity"), tPrevMomentumWS);

            Plato::ScalarMultiVectorT<PrevPressScalarT> tPrevPressWS("Previous Pressure Workset", tNumCells, mNumPressDofsPerCell);
            Plato::workset_state_scalar_scalar<mNumPressDofsPerNode, mNumNodesPerCell>
                (tDomain, mPressStateEntryOrdinal, aState.get("previous pressure"), tPrevPressWS);

            Plato::ScalarMultiVectorT<PrevTempScalarT> tPrevTempWS("Previous Temperature Workset", tNumCells, mNumTempDofsPerCell);
            Plato::workset_state_scalar_scalar<mNumTempDofsPerNode, mNumNodesPerCell>
                (tDomain, mTempStateEntryOrdinal, aState.get("previous temperature"), tPrevTempWS);

            Plato::ScalarMultiVectorT<ControlScalarT> tControlWS("Control Workset", tNumCells, mNumNodesPerCell);
            Plato::workset_control_scalar_scalar<mNumNodesPerCell>
                (tDomain, mControlEntryOrdinal, aState.get("current controls"), tControlWS);

            Plato::ScalarArray3DT<ConfigScalarT> tConfigWS("Config Workset", tNumCells, mNumNodesPerCell, mNumSpatialDims);
            Plato::workset_config_scalar<mNumSpatialDims, mNumNodesPerCell>(tDomain, mNodeCoordinate, tConfigWS);

            Plato::ScalarMultiVectorT<ResultScalarT> tResidualWS("Residual Workset", tNumCells, mNumVelDofsPerCell);
            //mResidualFunctions.at(tName)->evaluate( tStateWS, tStateDotWS, tStateDotDotWS, tControlWS, tConfigWS, tResidualWS, aTimeStep, aCurrentTime );

            Plato::assemble_residual<mNumNodesPerCell, mNumVelDofsPerNode>
                (tDomain, mVelStateEntryOrdinal, tResidualWS, tReturnValue);
        }

        // external force contribution
        {
            auto tNumCells = mSpatialModel.Mesh.nverts();

            Plato::ScalarMultiVectorT<PrevVelScalarT> tVelPredictorWS("Velocity Predictor Workset", tNumCells, mNumVelDofsPerCell);
            Plato::workset_state_scalar_scalar<mNumVelDofsPerNode, mNumNodesPerCell>
                (tNumCells, mVelStateEntryOrdinal, aState.get("velocity predictor"), tVelPredictorWS);

            Plato::ScalarMultiVectorT<PrevVelScalarT> tPrevMomentumWS("Previous Velocity Workset", tNumCells, mNumVelDofsPerCell);
            Plato::workset_state_scalar_scalar<mNumVelDofsPerNode, mNumNodesPerCell>
                (tNumCells, mVelStateEntryOrdinal, aState.get("previous velocity"), tPrevMomentumWS);

            Plato::ScalarMultiVectorT<PrevPressScalarT> tPrevPressWS("Previous Pressure Workset", tNumCells, mNumPressDofsPerCell);
            Plato::workset_state_scalar_scalar<mNumPressDofsPerNode, mNumNodesPerCell>
                (tNumCells, mPressStateEntryOrdinal, aState.get("previous pressure"), tPrevPressWS);

            Plato::ScalarMultiVectorT<PrevTempScalarT> tPrevTempWS("Previous Temperature Workset", tNumCells, mNumTempDofsPerCell);
            Plato::workset_state_scalar_scalar<mNumTempDofsPerNode, mNumNodesPerCell>
                (tNumCells, mTempStateEntryOrdinal, aState.get("previous temperature"), tPrevTempWS);

            Plato::ScalarMultiVectorT<ControlScalarT> tControlWS("Control Workset", tNumCells, mNumNodesPerCell);
            Plato::workset_control_scalar_scalar<mNumNodesPerCell>
                (tNumCells, mControlEntryOrdinal, aState.get("current controls"), tControlWS);

            Plato::ScalarArray3DT<ConfigScalarT> tConfigWS("Config Workset", tNumCells, mNumNodesPerCell, mNumSpatialDims);
            Plato::workset_config_scalar<mNumSpatialDims, mNumNodesPerCell>(tNumCells, mNodeCoordinate, tConfigWS);

            Plato::ScalarMultiVectorT<ResultScalarT> tResidualWS("Residual Workset", tNumCells, mNumVelDofsPerCell);
            //mResidualFunctions.begin()->evaluate_boundary(mSpatialModel, tStateWS, tStateDotWS, tStateDotDotWS, tControlWS, tConfigWS, tResidualWS, aTimeStep, aCurrentTime );

            Plato::assemble_residual<mNumNodesPerCell, mNumVelDofsPerNode>
                (tNumCells, mVelStateEntryOrdinal, tResidualWS, tReturnValue);
        }

        return tReturnValue;
    }

    Teuchos::RCP<Plato::CrsMatrixType>
    gradient_x(Plato::Hyperbolic::FluidMechanics::State& aState) const
    {
        using ControlScalarT   = typename Residual::ControlScalarType;
        using ConfigScalarT    = typename Residual::ConfigScalarType;
        using ResultScalarT    = typename Residual::ResultScalarType;
        using VelPredScalarT   = typename Residual::MomentumPredictorScalarType;
        using PrevVelScalarT   = typename Residual::PreviousMomentumScalarType;
        using PrevTempScalarT  = typename Residual::PreviousEnergyScalarType;
        using PrevPressScalarT = typename Residual::PreviousMassScalarType;

        // create return matrix
        //
        auto tMesh = mSpatialModel.Mesh;
        Teuchos::RCP<Plato::CrsMatrixType> tJacobian =
            Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumSpatialDims, mNumVelDofsPerNode>(&tMesh);
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            auto tName     = tDomain.getDomainName();
        }

        return tJacobian;
    }
};

}
// namespace Momentum

}
// namespace Hyperbolic

namespace FluidMechanics
{

}
// namespace FluidMechanics

}
//namespace Plato
