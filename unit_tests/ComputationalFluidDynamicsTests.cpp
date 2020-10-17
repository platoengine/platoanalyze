/*
 * ComputationalFluidDynamicsTests.cpp
 *
 *  Created on: Oct 13, 2020
 */

#include "Teuchos_UnitTestHarness.hpp"
#include "PlatoTestHelpers.hpp"

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

struct States
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
    Plato::ScalarVector get(const std::string& aTag) const
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

  using ControlScalarType           = FadType;
  using ConfigScalarType            = Plato::Scalar;
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


template<typename SimplexPhysicsT, typename EvaluationT>
struct WorkSets
{
private:
    static constexpr auto mNumControls            = SimplexPhysicsT::mNumControl;             /*!< number of design variable fields */
    static constexpr auto mNumSpatialDims         = SimplexPhysicsT::mNumSpatialDims;         /*!< number of spatial dimensions */
    static constexpr auto mNumNodesPerCell        = SimplexPhysicsT::mNumNodesPerCell;        /*!< number of nodes per cell */
    static constexpr auto mNumMassDofsPerCell     = SimplexPhysicsT::mNumMassDofsPerCell;     /*!< number of mass dofs per cell */
    static constexpr auto mNumEnergyDofsPerCell   = SimplexPhysicsT::mNumEnergyDofsPerCell;   /*!< number of energy dofs per cell */
    static constexpr auto mNumMomentumDofsPerCell = SimplexPhysicsT::mNumMomentumDofsPerCell; /*!< number of momentum dofs per cell */

    Plato::OrdinalType mNumCells;

    Plato::ScalarArray3DT<typename EvaluationT::ConfigScalarType> mConfiguration;
    Plato::ScalarMultiVectorT<typename EvaluationT::ResultScalarType> mResult;
    Plato::ScalarMultiVectorT<typename EvaluationT::ControlScalarType> mControls;
    Plato::ScalarMultiVectorT<typename EvaluationT::CurrentMassScalarType> mCurrentMass;
    Plato::ScalarMultiVectorT<typename EvaluationT::CurrentEnergyScalarType> mCurrentEnergy;
    Plato::ScalarMultiVectorT<typename EvaluationT::CurrentMomentumScalarType> mCurrentMomentum;
    Plato::ScalarMultiVectorT<typename EvaluationT::PreviousMassScalarType> mPreviousMass;
    Plato::ScalarMultiVectorT<typename EvaluationT::PreviousEnergyScalarType> mPreviousEnergy;
    Plato::ScalarMultiVectorT<typename EvaluationT::PreviousMomentumScalarType> mPreviousMomentum;
    Plato::ScalarMultiVectorT<typename EvaluationT::MomentumPredictorScalarType> mMomentumPredictor;

public:
    explicit WorkSets(const Plato::OrdinalType& aNumCells) :
        mNumCells(aNumCells),
        mConfiguration("Configuration Workset", aNumCells, mNumNodesPerCell, mNumSpatialDims),
        mResult("Result Workset", aNumCells, mNumMomentumDofsPerCell),
        mControls("Control Workset", aNumCells, mNumNodesPerCell),
        mCurrentMass("Current Mass Workset", aNumCells, mNumMassDofsPerCell),
        mCurrentEnergy("Current Energy Workset", aNumCells, mNumEnergyDofsPerCell),
        mCurrentMomentum("Current Momentum Workset", aNumCells, mNumMomentumDofsPerCell),
        mPreviousMass("Previous Mass Workset", aNumCells, mNumMassDofsPerCell),
        mPreviousEnergy("Previous Energy Workset", aNumCells, mNumEnergyDofsPerCell),
        mPreviousMomentum("Previous Momentum Workset", aNumCells, mNumMomentumDofsPerCell),
        mMomentumPredictor("Momentum Predictor Workset", aNumCells, mNumMomentumDofsPerCell)
    {}

    decltype(mNumCells) numCells() const
    {
        return mNumCells;
    }

    decltype(mConfiguration) configuration()
    {
        return mConfiguration;
    }

    decltype(mResult) result()
    {
        return mResult;
    }

    decltype(mControls) controls()
    {
        return mControls;
    }

    decltype(mCurrentMass) currentMass()
    {
        return mCurrentMass;
    }

    decltype(mCurrentEnergy) currentEnergy()
    {
        return mCurrentEnergy;
    }

    decltype(mCurrentMomentum) currentMomentum()
    {
        return mCurrentMomentum;
    }

    decltype(mPreviousMass) previousMass()
    {
        return mPreviousMass;
    }

    decltype(mPreviousEnergy) previousEnergy()
    {
        return mPreviousEnergy;
    }

    decltype(mPreviousMomentum) previousMomentum()
    {
        return mPreviousMomentum;
    }

    decltype(mMomentumPredictor) momentumPredictor()
    {
        return mMomentumPredictor;
    }
};

template<typename SimplexPhysicsT, typename EvaluationT>
class AbstractVectorFunction
{
public:
    AbstractVectorFunction(){}
    virtual ~AbstractVectorFunction(){}

    virtual void evaluate(Plato::Hyperbolic::FluidMechanics::WorkSets<SimplexPhysicsT, EvaluationT>& aWorkSets) const = 0;
    virtual void evaluate_boundary(Plato::Hyperbolic::FluidMechanics::WorkSets<SimplexPhysicsT, EvaluationT>& aWorkSets) const = 0;
};
// class AbstractVectorFunction

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
    static constexpr auto mNumControlPerNode      = PhysicsT::SimplexT::mNumControl;             /*!< number of design variable fields */
    static constexpr auto mNumSpatialDims         = PhysicsT::SimplexT::mNumSpatialDims;         /*!< number of spatial dimensions */
    static constexpr auto mNumNodesPerCell        = PhysicsT::SimplexT::mNumNodesPerCell;        /*!< number of nodes per cell */
    static constexpr auto mNumMassDofsPerCell     = PhysicsT::SimplexT::mNumMassDofsPerCell;     /*!< number of mass dofs per cell */
    static constexpr auto mNumEnergyDofsPerCell   = PhysicsT::SimplexT::mNumEnergyDofsPerCell;   /*!< number of energy dofs per cell */
    static constexpr auto mNumMomentumDofsPerCell = PhysicsT::SimplexT::mNumMomentumDofsPerCell; /*!< number of momentum dofs per cell */
    static constexpr auto mNumMassDofsPerNode     = PhysicsT::SimplexT::mNumMassDofsPerNode;     /*!< number of mass dofs per node */
    static constexpr auto mNumEnergyDofsPerNode   = PhysicsT::SimplexT::mNumEnergyDofsPerNode;   /*!< number of energy dofs per node */
    static constexpr auto mNumMomentumDofsPerNode = PhysicsT::SimplexT::mNumMomentumDofsPerNode; /*!< number of momentum dofs per node */
    static constexpr auto mNumConfigDofsPerCell   = mNumSpatialDims * mNumNodesPerCell;          /*!< number of configuration degrees of freedom per cell */

    // forward automatic differentiation types
    using Residual         = typename Plato::Hyperbolic::FluidMechanics::Evaluation<typename PhysicsT::SimplexT>::Residual;
    using GradConfig       = typename Plato::Hyperbolic::FluidMechanics::Evaluation<typename PhysicsT::SimplexT>::GradConfig;
    using GradControl      = typename Plato::Hyperbolic::FluidMechanics::Evaluation<typename PhysicsT::SimplexT>::GradControl;
    using GradCurrMass     = typename Plato::Hyperbolic::FluidMechanics::Evaluation<typename PhysicsT::SimplexT>::GradMassCurr;
    using GradPrevMass     = typename Plato::Hyperbolic::FluidMechanics::Evaluation<typename PhysicsT::SimplexT>::GradMassPrev;
    using GradCurrEnergy   = typename Plato::Hyperbolic::FluidMechanics::Evaluation<typename PhysicsT::SimplexT>::GradEnergyCurr;
    using GradPrevEnergy   = typename Plato::Hyperbolic::FluidMechanics::Evaluation<typename PhysicsT::SimplexT>::GradEnergyPrev;
    using GradCurrMomentum = typename Plato::Hyperbolic::FluidMechanics::Evaluation<typename PhysicsT::SimplexT>::GradMomentumCurr;
    using GradPrevMomentum = typename Plato::Hyperbolic::FluidMechanics::Evaluation<typename PhysicsT::SimplexT>::GradMomentumPrev;
    using GradMomentumPred = typename Plato::Hyperbolic::FluidMechanics::Evaluation<typename PhysicsT::SimplexT>::GradMomentumPred;

    // element residual functions
    using ResidualFunc         = std::shared_ptr<Plato::Hyperbolic::FluidMechanics::AbstractVectorFunction<typename PhysicsT::SimplexT, Residual>>;
    using GradConfigFunc       = std::shared_ptr<Plato::Hyperbolic::FluidMechanics::AbstractVectorFunction<typename PhysicsT::SimplexT, GradConfig>>;
    using GradControlFunc      = std::shared_ptr<Plato::Hyperbolic::FluidMechanics::AbstractVectorFunction<typename PhysicsT::SimplexT, GradControl>>;
    using GradCurrMassFunc     = std::shared_ptr<Plato::Hyperbolic::FluidMechanics::AbstractVectorFunction<typename PhysicsT::SimplexT, GradCurrMass>>;
    using GradPrevMassFunc     = std::shared_ptr<Plato::Hyperbolic::FluidMechanics::AbstractVectorFunction<typename PhysicsT::SimplexT, GradPrevMass>>;
    using GradCurrEnergyFunc   = std::shared_ptr<Plato::Hyperbolic::FluidMechanics::AbstractVectorFunction<typename PhysicsT::SimplexT, GradCurrEnergy>>;
    using GradPrevEnergyFunc   = std::shared_ptr<Plato::Hyperbolic::FluidMechanics::AbstractVectorFunction<typename PhysicsT::SimplexT, GradPrevEnergy>>;
    using GradCurrMomentumFunc = std::shared_ptr<Plato::Hyperbolic::FluidMechanics::AbstractVectorFunction<typename PhysicsT::SimplexT, GradCurrMomentum>>;
    using GradPrevMomentumFunc = std::shared_ptr<Plato::Hyperbolic::FluidMechanics::AbstractVectorFunction<typename PhysicsT::SimplexT, GradPrevMomentum>>;
    using GradMomentumPredFunc = std::shared_ptr<Plato::Hyperbolic::FluidMechanics::AbstractVectorFunction<typename PhysicsT::SimplexT, GradMomentumPred>>;

    std::unordered_map<std::string, ResidualFunc>         mResidualFuncs;
    std::unordered_map<std::string, GradConfigFunc>       mGradConfigFuncs;
    std::unordered_map<std::string, GradControlFunc>      mGradControlFuncs;
    std::unordered_map<std::string, GradCurrMassFunc>     mGradCurrMassFuncs;
    std::unordered_map<std::string, GradPrevMassFunc>     mGradPrevMassFuncs;
    std::unordered_map<std::string, GradCurrEnergyFunc>   mGradCurrEnergyFuncs;
    std::unordered_map<std::string, GradPrevEnergyFunc>   mGradPrevEnergyFuncs;
    std::unordered_map<std::string, GradCurrMomentumFunc> mGradCurrMomentumFuncs;
    std::unordered_map<std::string, GradPrevMomentumFunc> mGradPrevMomentumFuncs;
    std::unordered_map<std::string, GradMomentumPredFunc> mGradMomentumPredFuncs;

    Plato::DataMap& mDataMap;
    const Plato::SpatialModel& mSpatialModel;

    Plato::NodeCoordinate<mNumSpatialDims> mNodeCoordinate; /*!< node coordinates metadata */

    Plato::VectorEntryOrdinal<mNumSpatialDims, mNumSpatialDims>         mConfigEntryOrdinal;     /*!< configuration local-to-global ID map */
    Plato::VectorEntryOrdinal<mNumSpatialDims, mNumControlPerNode>      mControlEntryOrdinal;    /*!< control local-to-global ID map */
    Plato::VectorEntryOrdinal<mNumSpatialDims, mNumMassDofsPerCell>     mPressStateEntryOrdinal; /*!< mass state local-to-global ID map */
    Plato::VectorEntryOrdinal<mNumSpatialDims, mNumEnergyDofsPerCell>   mTempStateEntryOrdinal;  /*!< energy state local-to-global ID map */
    Plato::VectorEntryOrdinal<mNumSpatialDims, mNumMomentumDofsPerCell> mVelStateEntryOrdinal;   /*!< momentum state local-to-global ID map */

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
        return (tNumNodes * mNumMomentumDofsPerCell);
    }

    Plato::ScalarVector
    value(const Plato::Hyperbolic::FluidMechanics::States& aStates) const
    {
        auto tLength = this->size();
        Plato::ScalarVector tReturnValue("Assembled Residual", tLength);

        // internal force contribution
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            Plato::Hyperbolic::FluidMechanics::WorkSets<PhysicsT, Residual> tWorkSets(tNumCells);
            this->setWorkSets(tDomain, aStates, tWorkSets);

            auto tName = tDomain.getDomainName();
            mResidualFuncs.at(tName)->evaluate(tWorkSets);

            auto tResidualWS = tWorkSets.result();
            Plato::assemble_residual<mNumNodesPerCell, mNumMomentumDofsPerNode>
                (tDomain, mVelStateEntryOrdinal, tResidualWS, tReturnValue);
        }

        // external force contribution
        {
            auto tNumCells = mSpatialModel.Mesh.nverts();
            Plato::Hyperbolic::FluidMechanics::WorkSets<PhysicsT, Residual> tWorkSets(tNumCells);
            this->setWorkSets(aStates, tWorkSets);

            mResidualFuncs.begin()->evaluate_boundary(tWorkSets);

            auto tResidualWS = tWorkSets.result();
            Plato::assemble_residual<mNumNodesPerCell, mNumMomentumDofsPerNode>
                (tNumCells, mVelStateEntryOrdinal, tResidualWS, tReturnValue);
        }

        return tReturnValue;
    }

    Teuchos::RCP<Plato::CrsMatrixType>
    gradient_x(Plato::Hyperbolic::FluidMechanics::States& aStates) const
    {
        // create return matrix
        auto tMesh = mSpatialModel.Mesh;
        Teuchos::RCP<Plato::CrsMatrixType> tJacobian =
            Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumSpatialDims, mNumMomentumDofsPerNode>(&tMesh);

        // internal force contribution
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            Plato::Hyperbolic::FluidMechanics::WorkSets<PhysicsT, GradConfig> tWorkSets(tNumCells);
            this->setWorkSets(tDomain, aStates, tWorkSets);

            auto tName = tDomain.getDomainName();
            mGradConfigFuncs.at(tName)->evaluate(tWorkSets);

            auto tJacobianWS = tWorkSets.result();
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumSpatialDims, mNumMomentumDofsPerNode> tJacobianEntryOrdinal(tJacobian, &tMesh);
            Plato::assemble_transpose_jacobian(tDomain, mNumMomentumDofsPerCell, mNumConfigDofsPerCell, tJacobianEntryOrdinal, tJacobianWS, tJacobian->entries());
        }

        // external force contribution
        {
            auto tNumCells = mSpatialModel.Mesh.nverts();
            Plato::Hyperbolic::FluidMechanics::WorkSets<PhysicsT, GradConfig> tWorkSets(tNumCells);
            this->setWorkSets(aStates, tWorkSets);

            mGradConfigFuncs.begin()->evaluate_boundary(tWorkSets);

            auto tJacobianWS = tWorkSets.result();
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumSpatialDims, mNumMomentumDofsPerNode> tJacobianEntryOrdinal(tJacobian, &tMesh);
            Plato::assemble_transpose_jacobian(tNumCells, mNumMomentumDofsPerCell, mNumConfigDofsPerCell, tJacobianEntryOrdinal, tJacobianWS, tJacobian->entries());
        }

        return tJacobian;
    }

    Teuchos::RCP<Plato::CrsMatrixType>
    gradient_z(Plato::Hyperbolic::FluidMechanics::States& aStates) const
    {
        // create return matrix
        auto tMesh = mSpatialModel.Mesh;
        Teuchos::RCP<Plato::CrsMatrixType> tJacobian =
            Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumControlPerNode, mNumMomentumDofsPerNode>( &tMesh );

        // internal force contribution
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            Plato::Hyperbolic::FluidMechanics::WorkSets<PhysicsT, GradControl> tWorkSets(tNumCells);
            this->setWorkSets(tDomain, aStates, tWorkSets);

            auto tName = tDomain.getDomainName();
            mGradControlFuncs.at(tName)->evaluate(tWorkSets);

            auto tJacobianWS = tWorkSets.result();
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumControlPerNode, mNumMomentumDofsPerNode> tJacobianEntryOrdinal(tJacobian, &tMesh);
            Plato::assemble_transpose_jacobian(tDomain, mNumMomentumDofsPerCell, mNumNodesPerCell, tJacobianEntryOrdinal, tJacobianWS, tJacobian->entries());
        }

        // external force contribution
        {
            auto tNumCells = mSpatialModel.Mesh.nverts();
            Plato::Hyperbolic::FluidMechanics::WorkSets<PhysicsT, GradControl> tWorkSets(tNumCells);
            this->setWorkSets(aStates, tWorkSets);

            mGradControlFuncs.begin()->evaluate_boundary(tWorkSets);

            auto tJacobianWS = tWorkSets.result();
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumControlPerNode, mNumMomentumDofsPerNode> tJacobianEntryOrdinal(tJacobian, &tMesh);
            Plato::assemble_transpose_jacobian(tNumCells, mNumMomentumDofsPerCell, mNumNodesPerCell, tJacobianEntryOrdinal, tJacobianWS, tJacobian->entries());
        }

        return tJacobian;
    }

    Teuchos::RCP<Plato::CrsMatrixType>
    gradient_du(Plato::Hyperbolic::FluidMechanics::States& aStates) const
    {
        // create return matrix
        auto tMesh = mSpatialModel.Mesh;
        Teuchos::RCP<Plato::CrsMatrixType> tJacobian =
            Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumMomentumDofsPerNode, mNumMomentumDofsPerNode>( &tMesh );

        // internal force contribution
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            Plato::Hyperbolic::FluidMechanics::WorkSets<PhysicsT, GradMomentumPred> tWorkSets(tNumCells);
            this->setWorkSets(tDomain, aStates, tWorkSets);

            auto tName = tDomain.getDomainName();
            mGradMomentumPredFuncs.at(tName)->evaluate(tWorkSets);

            auto tJacobianWS = tWorkSets.result();
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumMomentumDofsPerNode, mNumMomentumDofsPerNode> tJacobianEntryOrdinal(tJacobian, &tMesh);
            Plato::assemble_transpose_jacobian(tDomain, mNumMomentumDofsPerCell, mNumMomentumDofsPerCell, tJacobianEntryOrdinal, tJacobianWS, tJacobian->entries());
        }

        // external force contribution
        {
            auto tNumCells = mSpatialModel.Mesh.nverts();
            Plato::Hyperbolic::FluidMechanics::WorkSets<PhysicsT, GradMomentumPred> tWorkSets(tNumCells);
            this->setWorkSets(aStates, tWorkSets);

            mGradMomentumPredFuncs.begin()->evaluate_boundary(tWorkSets);

            auto tJacobianWS = tWorkSets.result();
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumMomentumDofsPerNode, mNumMomentumDofsPerNode> tJacobianEntryOrdinal(tJacobian, &tMesh);
            Plato::assemble_transpose_jacobian(tNumCells, mNumMomentumDofsPerCell, mNumMomentumDofsPerCell, tJacobianEntryOrdinal, tJacobianWS, tJacobian->entries());
        }

        return tJacobian;
    }

    Teuchos::RCP<Plato::CrsMatrixType>
    gradient_up(Plato::Hyperbolic::FluidMechanics::States& aStates) const
    {
        // create return matrix
        auto tMesh = mSpatialModel.Mesh;
        Teuchos::RCP<Plato::CrsMatrixType> tJacobian =
            Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumMomentumDofsPerNode, mNumMomentumDofsPerNode>( &tMesh );

        // internal force contribution
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            Plato::Hyperbolic::FluidMechanics::WorkSets<PhysicsT, GradPrevMomentum> tWorkSets(tNumCells);
            this->setWorkSets(tDomain, aStates, tWorkSets);

            auto tName = tDomain.getDomainName();
            mGradPrevMomentumFuncs.at(tName)->evaluate(tWorkSets);

            auto tJacobianWS = tWorkSets.result();
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumMomentumDofsPerNode, mNumMomentumDofsPerNode> tJacobianEntryOrdinal(tJacobian, &tMesh);
            Plato::assemble_transpose_jacobian(tDomain, mNumMomentumDofsPerCell, mNumMomentumDofsPerCell, tJacobianEntryOrdinal, tJacobianWS, tJacobian->entries());
        }

        // external force contribution
        {
            auto tNumCells = mSpatialModel.Mesh.nverts();
            Plato::Hyperbolic::FluidMechanics::WorkSets<PhysicsT, GradPrevMomentum> tWorkSets(tNumCells);
            this->setWorkSets(aStates, tWorkSets);

            mGradPrevMomentumFuncs.begin()->evaluate_boundary(tWorkSets);

            auto tJacobianWS = tWorkSets.result();
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumMomentumDofsPerNode, mNumMomentumDofsPerNode> tJacobianEntryOrdinal(tJacobian, &tMesh);
            Plato::assemble_transpose_jacobian(tNumCells, mNumMomentumDofsPerCell, mNumMomentumDofsPerCell, tJacobianEntryOrdinal, tJacobianWS, tJacobian->entries());
        }

        return tJacobian;
    }

    Teuchos::RCP<Plato::CrsMatrixType>
    gradient_mp(Plato::Hyperbolic::FluidMechanics::States& aStates) const
    {
        // create return matrix
        auto tMesh = mSpatialModel.Mesh;
        Teuchos::RCP<Plato::CrsMatrixType> tJacobian =
            Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumMassDofsPerNode, mNumMomentumDofsPerNode>( &tMesh );

        // internal force contribution
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            Plato::Hyperbolic::FluidMechanics::WorkSets<PhysicsT, GradPrevMass> tWorkSets(tNumCells);
            this->setWorkSets(tDomain, aStates, tWorkSets);

            auto tName = tDomain.getDomainName();
            mGradPrevMassFuncs.at(tName)->evaluate(tWorkSets);

            auto tJacobianWS = tWorkSets.result();
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumMassDofsPerNode, mNumMomentumDofsPerNode> tJacobianEntryOrdinal(tJacobian, &tMesh);
            Plato::assemble_transpose_jacobian(tDomain, mNumMomentumDofsPerCell, mNumMassDofsPerCell, tJacobianEntryOrdinal, tJacobianWS, tJacobian->entries());
        }

        // external force contribution
        {
            auto tNumCells = mSpatialModel.Mesh.nverts();
            Plato::Hyperbolic::FluidMechanics::WorkSets<PhysicsT, GradPrevMass> tWorkSets(tNumCells);
            this->setWorkSets(aStates, tWorkSets);

            mGradPrevMassFuncs.begin()->evaluate_boundary(tWorkSets);

            auto tJacobianWS = tWorkSets.result();
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumMassDofsPerNode, mNumMomentumDofsPerNode> tJacobianEntryOrdinal(tJacobian, &tMesh);
            Plato::assemble_transpose_jacobian(tNumCells, mNumMomentumDofsPerCell, mNumMassDofsPerCell, tJacobianEntryOrdinal, tJacobianWS, tJacobian->entries());
        }

        return tJacobian;
    }

    Teuchos::RCP<Plato::CrsMatrixType>
    gradient_ep(Plato::Hyperbolic::FluidMechanics::States& aStates) const
    {
        // create return matrix
        auto tMesh = mSpatialModel.Mesh;
        Teuchos::RCP<Plato::CrsMatrixType> tJacobian =
            Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumEnergyDofsPerNode, mNumMomentumDofsPerNode>( &tMesh );

        // internal force contribution
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            Plato::Hyperbolic::FluidMechanics::WorkSets<PhysicsT, GradPrevEnergy> tWorkSets(tNumCells);
            this->setWorkSets(tDomain, aStates, tWorkSets);

            auto tName = tDomain.getDomainName();
            mGradPrevEnergyFuncs.at(tName)->evaluate(tWorkSets);

            auto tJacobianWS = tWorkSets.result();
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumEnergyDofsPerNode, mNumMomentumDofsPerNode> tJacobianEntryOrdinal(tJacobian, &tMesh);
            Plato::assemble_transpose_jacobian(tDomain, mNumMomentumDofsPerCell, mNumEnergyDofsPerCell, tJacobianEntryOrdinal, tJacobianWS, tJacobian->entries());
        }

        // external force contribution
        {
            auto tNumCells = mSpatialModel.Mesh.nverts();
            Plato::Hyperbolic::FluidMechanics::WorkSets<PhysicsT, GradPrevEnergy> tWorkSets(tNumCells);
            this->setWorkSets(aStates, tWorkSets);

            mGradPrevEnergyFuncs.begin()->evaluate_boundary(tWorkSets);

            auto tJacobianWS = tWorkSets.result();
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumEnergyDofsPerNode, mNumMomentumDofsPerNode> tJacobianEntryOrdinal(tJacobian, &tMesh);
            Plato::assemble_transpose_jacobian(tNumCells, mNumMomentumDofsPerCell, mNumEnergyDofsPerCell, tJacobianEntryOrdinal, tJacobianWS, tJacobian->entries());
        }

        return tJacobian;
    }

    Teuchos::RCP<Plato::CrsMatrixType>
    gradient_uc(Plato::Hyperbolic::FluidMechanics::States& aStates) const
    {
        // create return matrix
        auto tMesh = mSpatialModel.Mesh;
        Teuchos::RCP<Plato::CrsMatrixType> tJacobian =
            Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumMomentumDofsPerNode, mNumMomentumDofsPerNode>( &tMesh );

        // internal force contribution
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            Plato::Hyperbolic::FluidMechanics::WorkSets<PhysicsT, GradCurrMomentum> tWorkSets(tNumCells);
            this->setWorkSets(tDomain, aStates, tWorkSets);

            auto tName = tDomain.getDomainName();
            mGradCurrMomentumFuncs.at(tName)->evaluate(tWorkSets);

            auto tJacobianWS = tWorkSets.result();
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumMomentumDofsPerNode, mNumMomentumDofsPerNode> tJacobianEntryOrdinal(tJacobian, &tMesh);
            Plato::assemble_transpose_jacobian(tDomain, mNumMomentumDofsPerCell, mNumMomentumDofsPerCell, tJacobianEntryOrdinal, tJacobianWS, tJacobian->entries());
        }

        // external force contribution
        {
            auto tNumCells = mSpatialModel.Mesh.nverts();
            Plato::Hyperbolic::FluidMechanics::WorkSets<PhysicsT, GradCurrMomentum> tWorkSets(tNumCells);
            this->setWorkSets(aStates, tWorkSets);

            mGradCurrMomentumFuncs.begin()->evaluate_boundary(tWorkSets);

            auto tJacobianWS = tWorkSets.result();
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumMomentumDofsPerNode, mNumMomentumDofsPerNode> tJacobianEntryOrdinal(tJacobian, &tMesh);
            Plato::assemble_transpose_jacobian(tNumCells, mNumMomentumDofsPerCell, mNumMomentumDofsPerCell, tJacobianEntryOrdinal, tJacobianWS, tJacobian->entries());
        }

        return tJacobian;
    }

    Teuchos::RCP<Plato::CrsMatrixType>
    gradient_mc(Plato::Hyperbolic::FluidMechanics::States& aStates) const
    {
        // create return matrix
        auto tMesh = mSpatialModel.Mesh;
        Teuchos::RCP<Plato::CrsMatrixType> tJacobian =
            Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumMassDofsPerNode, mNumMomentumDofsPerNode>( &tMesh );

        // internal force contribution
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            Plato::Hyperbolic::FluidMechanics::WorkSets<PhysicsT, GradCurrMass> tWorkSets(tNumCells);
            this->setWorkSets(tDomain, aStates, tWorkSets);

            auto tName = tDomain.getDomainName();
            mGradCurrMassFuncs.at(tName)->evaluate(tWorkSets);

            auto tJacobianWS = tWorkSets.result();
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumMassDofsPerNode, mNumMomentumDofsPerNode> tJacobianEntryOrdinal(tJacobian, &tMesh);
            Plato::assemble_transpose_jacobian(tDomain, mNumMomentumDofsPerCell, mNumMassDofsPerCell, tJacobianEntryOrdinal, tJacobianWS, tJacobian->entries());
        }

        // external force contribution
        {
            auto tNumCells = mSpatialModel.Mesh.nverts();
            Plato::Hyperbolic::FluidMechanics::WorkSets<PhysicsT, GradCurrMass> tWorkSets(tNumCells);
            this->setWorkSets(aStates, tWorkSets);

            mGradCurrMassFuncs.begin()->evaluate_boundary(tWorkSets);

            auto tJacobianWS = tWorkSets.result();
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumMassDofsPerNode, mNumMomentumDofsPerNode> tJacobianEntryOrdinal(tJacobian, &tMesh);
            Plato::assemble_transpose_jacobian(tNumCells, mNumMomentumDofsPerCell, mNumMassDofsPerCell, tJacobianEntryOrdinal, tJacobianWS, tJacobian->entries());
        }

        return tJacobian;
    }

    Teuchos::RCP<Plato::CrsMatrixType>
    gradient_ec(Plato::Hyperbolic::FluidMechanics::States& aStates) const
    {
        // create return matrix
        auto tMesh = mSpatialModel.Mesh;
        Teuchos::RCP<Plato::CrsMatrixType> tJacobian =
            Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumEnergyDofsPerNode, mNumMomentumDofsPerNode>( &tMesh );

        // internal force contribution
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            Plato::Hyperbolic::FluidMechanics::WorkSets<PhysicsT, GradCurrEnergy> tWorkSets(tNumCells);
            this->setWorkSets(tDomain, aStates, tWorkSets);

            auto tName = tDomain.getDomainName();
            mGradCurrEnergyFuncs.at(tName)->evaluate(tWorkSets);

            auto tJacobianWS = tWorkSets.result();
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumEnergyDofsPerNode, mNumMomentumDofsPerNode> tJacobianEntryOrdinal(tJacobian, &tMesh);
            Plato::assemble_transpose_jacobian(tDomain, mNumMomentumDofsPerCell, mNumEnergyDofsPerCell, tJacobianEntryOrdinal, tJacobianWS, tJacobian->entries());
        }

        // external force contribution
        {
            auto tNumCells = mSpatialModel.Mesh.nverts();
            Plato::Hyperbolic::FluidMechanics::WorkSets<PhysicsT, GradCurrEnergy> tWorkSets(tNumCells);
            this->setWorkSets(aStates, tWorkSets);

            mGradCurrEnergyFuncs.begin()->evaluate_boundary(tWorkSets);

            auto tJacobianWS = tWorkSets.result();
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumEnergyDofsPerNode, mNumMomentumDofsPerNode> tJacobianEntryOrdinal(tJacobian, &tMesh);
            Plato::assemble_transpose_jacobian(tNumCells, mNumMomentumDofsPerCell, mNumEnergyDofsPerCell, tJacobianEntryOrdinal, tJacobianWS, tJacobian->entries());
        }

        return tJacobian;
    }

private:
    template<typename WorkSetT>
    void setWorkSets
    (const Plato::SpatialDomain& aDomain,
     const Plato::Hyperbolic::FluidMechanics::States& aState,
     WorkSetT& aWorkSets)
    {
        Plato::workset_state_scalar_scalar<mNumMomentumDofsPerNode, mNumNodesPerCell>
            (aDomain, mVelStateEntryOrdinal, aState.get("momentum predictor"), aWorkSets.momentumPredictor());

        Plato::workset_state_scalar_scalar<mNumMomentumDofsPerNode, mNumNodesPerCell>
            (aDomain, mVelStateEntryOrdinal, aState.get("previous momentum"), aWorkSets.previousMomentum());

        Plato::workset_state_scalar_scalar<mNumMassDofsPerNode, mNumNodesPerCell>
            (aDomain, mPressStateEntryOrdinal, aState.get("previous mass"), aWorkSets.previousMass());

        Plato::workset_state_scalar_scalar<mNumEnergyDofsPerNode, mNumNodesPerCell>
            (aDomain, mTempStateEntryOrdinal, aState.get("previous energy"), aWorkSets.previousEnergy());

        Plato::workset_control_scalar_scalar<mNumNodesPerCell>
            (aDomain, mControlEntryOrdinal, aState.get("controls"), aWorkSets.controls());

        Plato::workset_config_scalar<mNumSpatialDims, mNumNodesPerCell>
            (aDomain, mNodeCoordinate, aWorkSets.configuration());
    }

    template<typename WorkSetT>
    void setWorkSets
    (const Plato::Hyperbolic::FluidMechanics::States& aState,
     WorkSetT& aWorkSets)
    {
        auto tNumCells = aWorkSets.numCells();
        Plato::workset_state_scalar_scalar<mNumMomentumDofsPerNode, mNumNodesPerCell>
            (tNumCells, mVelStateEntryOrdinal, aState.get("momentum predictor"), aWorkSets.momentumPredictor());

        Plato::workset_state_scalar_scalar<mNumMomentumDofsPerNode, mNumNodesPerCell>
            (tNumCells, mVelStateEntryOrdinal, aState.get("previous momentum"), aWorkSets.previousMomentum());

        Plato::workset_state_scalar_scalar<mNumMassDofsPerNode, mNumNodesPerCell>
            (tNumCells, mPressStateEntryOrdinal, aState.get("previous mass"), aWorkSets.previousMass());

        Plato::workset_state_scalar_scalar<mNumEnergyDofsPerNode, mNumNodesPerCell>
            (tNumCells, mTempStateEntryOrdinal, aState.get("previous energy"), aWorkSets.previousEnergy());

        Plato::workset_control_scalar_scalar<mNumNodesPerCell>
            (tNumCells, mControlEntryOrdinal, aState.get("controls"), aWorkSets.controls());

        Plato::workset_config_scalar<mNumSpatialDims, mNumNodesPerCell>
            (tNumCells, mNodeCoordinate, aWorkSets.configuration());
    }

};
// class VectorFunction

}
// namespace FluidMechanics

}
// namespace Hyperbolic

}
//namespace Plato


namespace ComputationalFluidDynamicsTests
{

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, StateWS_Test)
{

}

}
