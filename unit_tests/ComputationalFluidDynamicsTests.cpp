/*
 * ComputationalFluidDynamicsTests.cpp
 *
 *  Created on: Oct 13, 2020
 */

#include "Teuchos_UnitTestHarness.hpp"
#include "PlatoTestHelpers.hpp"

#include <unordered_map>
#include <Omega_h_shape.hpp>

#include "BLAS1.hpp"
#include "Simplex.hpp"
#include "WorksetBase.hpp"
#include "SpatialModel.hpp"
#include "EssentialBCs.hpp"
#include "SimplexFadTypes.hpp"
#include "PlatoAbstractProblem.hpp"

#include "alg/PlatoSolverFactory.hpp"

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

template<Plato::OrdinalType SpaceDim, Plato::OrdinalType NumControls = 1>
class MomentumConservation : public Plato::SimplexFluidMechanics<SpaceDim, NumControls>
{
public:
    using SimplexT = Plato::SimplexFluidMechanics<SpaceDim, NumControls>;
    static constexpr auto mNumDofsPerNode = SimplexT::mNumMomentumDofsPerNode;
};

template<Plato::OrdinalType SpaceDim, Plato::OrdinalType NumControls = 1>
class MassConservation : public Plato::SimplexFluidMechanics<SpaceDim, NumControls>
{
public:
    using SimplexT = Plato::SimplexFluidMechanics<SpaceDim, NumControls>;
    static constexpr auto mNumDofsPerNode = SimplexT::mNumMassDofsPerNode;
};

template<Plato::OrdinalType SpaceDim, Plato::OrdinalType NumControls = 1>
class EnergyConservation : public Plato::SimplexFluidMechanics<SpaceDim, NumControls>
{
public:
    using SimplexT = Plato::SimplexFluidMechanics<SpaceDim, NumControls>;
    static constexpr auto mNumDofsPerNode = SimplexT::mNumEnergyDofsPerNode;
};

template<Plato::OrdinalType SpaceDim, Plato::OrdinalType NumControls = 1>
class IncompressibleFluids : public Plato::SimplexFluidMechanics<SpaceDim, NumControls>
{
public:
    using SimplexT = typename Plato::SimplexFluidMechanics<SpaceDim, NumControls>;
    static constexpr auto mNumSpatialDims = SpaceDim;
};

namespace Hyperbolic
{

struct Solution
{
private:
    std::unordered_map<std::string, Plato::ScalarMultiVector> mSolution;

public:
    void set(const std::string& aTag, const Plato::ScalarMultiVector& aData)
    {
        auto tLowerTag = Plato::tolower(aTag);
        mSolution[tLowerTag] = aData;
    }
    Plato::ScalarMultiVector get(const std::string& aTag) const
    {
        auto tLowerTag = Plato::tolower(aTag);
        auto tItr = mSolution.find(tLowerTag);
        if(tItr == mSolution.end())
        {
            THROWERR(std::string("Did not find array with tag '") + aTag + "' in solution map.")
        }
        return tItr->second;
    }
};

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
    std::unordered_map<std::string, Plato::Scalar> mScalars;
    std::unordered_map<std::string, Plato::ScalarVector> mStates;

public:
    Plato::Scalar getScalar(const std::string& aTag) const
    {
        auto tLowerTag = Plato::tolower(aTag);
        auto tItr = mScalars.find(tLowerTag);
        if(tItr == mScalars.end())
        {
            THROWERR(std::string("State scalar with tag '") + aTag + "' is not defined in state map.")
        }
        return tItr->second;
    }
    void setScalar(const std::string& aTag, const Plato::Scalar& aInput)
    {
        auto tLowerTag = Plato::tolower(aTag);
        mScalars[tLowerTag] = aInput;
    }

    Plato::ScalarVector getVector(const std::string& aTag) const
    {
        auto tLowerTag = Plato::tolower(aTag);
        auto tItr = mStates.find(tLowerTag);
        if(tItr == mStates.end())
        {
            THROWERR(std::string("State with tag '") + aTag + "' is not defined in state map.")
        }
        return tItr->second;
    }
    void setVector(const std::string& aTag, const Plato::ScalarVector& aInput)
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
    virtual void updateProblem(const Plato::Hyperbolic::FluidMechanics::States& aStates) const = 0;
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
            (aDomain, mVelStateEntryOrdinal, aState.getVector("momentum predictor"), aWorkSets.momentumPredictor());

        Plato::workset_state_scalar_scalar<mNumMomentumDofsPerNode, mNumNodesPerCell>
            (aDomain, mVelStateEntryOrdinal, aState.getVector("previous momentum"), aWorkSets.previousMomentum());

        Plato::workset_state_scalar_scalar<mNumMassDofsPerNode, mNumNodesPerCell>
            (aDomain, mPressStateEntryOrdinal, aState.getVector("previous mass"), aWorkSets.previousMass());

        Plato::workset_state_scalar_scalar<mNumEnergyDofsPerNode, mNumNodesPerCell>
            (aDomain, mTempStateEntryOrdinal, aState.getVector("previous energy"), aWorkSets.previousEnergy());

        Plato::workset_control_scalar_scalar<mNumNodesPerCell>
            (aDomain, mControlEntryOrdinal, aState.getVector("controls"), aWorkSets.controls());

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
            (tNumCells, mVelStateEntryOrdinal, aState.getVector("momentum predictor"), aWorkSets.momentumPredictor());

        Plato::workset_state_scalar_scalar<mNumMomentumDofsPerNode, mNumNodesPerCell>
            (tNumCells, mVelStateEntryOrdinal, aState.getVector("previous momentum"), aWorkSets.previousMomentum());

        Plato::workset_state_scalar_scalar<mNumMassDofsPerNode, mNumNodesPerCell>
            (tNumCells, mPressStateEntryOrdinal, aState.getVector("previous mass"), aWorkSets.previousMass());

        Plato::workset_state_scalar_scalar<mNumEnergyDofsPerNode, mNumNodesPerCell>
            (tNumCells, mTempStateEntryOrdinal, aState.getVector("previous energy"), aWorkSets.previousEnergy());

        Plato::workset_control_scalar_scalar<mNumNodesPerCell>
            (tNumCells, mControlEntryOrdinal, aState.getVector("controls"), aWorkSets.controls());

        Plato::workset_config_scalar<mNumSpatialDims, mNumNodesPerCell>
            (tNumCells, mNodeCoordinate, aWorkSets.configuration());
    }

};
// class VectorFunction

}
// namespace FluidMechanics

}
// namespace Hyperbolic

namespace cbs
{

template<Plato::OrdinalType SpaceDim>
inline Plato::ScalarVector
calculate_element_characteristic_size
(const Plato::OrdinalType & aNumCells,
 const Plato::NodeCoordinate<SpaceDim> & aNodeCoordinate)
{
    Omega_h::Few<Omega_h::Vector<SpaceDim>, SpaceDim + 1> tElementCoords;
    Plato::ScalarVector tElemCharacteristicSize("element size", aNumCells);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, aNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < SpaceDim + 1; tNodeIndex++)
        {
            for(Plato::OrdinalType tDimIndex = 0; tDimIndex < SpaceDim; tDimIndex++)
            {
                tElementCoords(tNodeIndex)(tDimIndex) = aNodeCoordinate(aCellOrdinal, tNodeIndex, tDimIndex);
            }
        }
        auto tSphere = Omega_h::get_inball(tElementCoords);
        tElemCharacteristicSize(aCellOrdinal) = static_cast<Plato::Scalar>(2.0) * tSphere.r;
    },"calculate characteristic element size");
    return tElemCharacteristicSize;
}

inline Plato::ScalarVector
calculate_artificial_compressibility
(const Plato::Hyperbolic::FluidMechanics::States& aStates,
 Plato::Scalar aEpsilonConstant = 0.5)
{
    auto tPrandtl = aStates.getScalar("prandtl");
    auto tReynolds = aStates.getScalar("reynolds");
    auto tElemSize = aStates.getVector("element size");
    auto tVelocity = aStates.getVector("current momentum");

    auto tLength = tVelocity.size();
    Plato::ScalarVector tArtificalCompress("artificial compressibility", tLength);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tLength), LAMBDA_EXPRESSION(const Plato::OrdinalType & aOrdinal)
    {
        // calculate velocities
        Plato::Scalar tConvectiveVelocity = tVelocity(aOrdinal) * tVelocity(aOrdinal);
        tConvectiveVelocity = sqrt(tConvectiveVelocity);
        auto tDiffusionVelocity = static_cast<Plato::Scalar>(1.0) / (tElemSize(aOrdinal) * tReynolds);
        auto tThermalVelocity = static_cast<Plato::Scalar>(1.0) / (tElemSize(aOrdinal) * tReynolds * tPrandtl);

        // calculate minimum artificial compressibility
        auto tArtificialCompressibility = (tConvectiveVelocity < tDiffusionVelocity) && (tConvectiveVelocity < tThermalVelocity)
            && (tConvectiveVelocity < aEpsilonConstant) ? tConvectiveVelocity : aEpsilonConstant;
        tArtificialCompressibility = (tDiffusionVelocity < tConvectiveVelocity ) && (tDiffusionVelocity < tThermalVelocity)
            && (tDiffusionVelocity < aEpsilonConstant) ? tDiffusionVelocity : tArtificialCompressibility;
        tArtificialCompressibility = (tThermalVelocity < tConvectiveVelocity ) && (tThermalVelocity < tDiffusionVelocity)
            && (tThermalVelocity < aEpsilonConstant) ? tThermalVelocity : tArtificialCompressibility;

        tArtificalCompress(aOrdinal) = tArtificialCompressibility;
    }, "calculate artificial compressibility");

    return tArtificalCompress;
}

inline Plato::ScalarVector
calculate_stable_time_step
(const Plato::Hyperbolic::FluidMechanics::States& aStates)
{
    auto tElemSize = aStates.getVector("element size");
    auto tVelocity = aStates.getVector("current momentum");
    auto tArtificialCompressibility = aStates.getVector("artificial compressibility");

    auto tLength = tVelocity.size();
    Plato::ScalarVector tTimeStep("time step", tLength);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tLength), LAMBDA_EXPRESSION(const Plato::OrdinalType & aOrdinal)
    {
        // calculate convective velocity
        Plato::Scalar tConvectiveVelocity = tVelocity(aOrdinal) * tVelocity(aOrdinal);
        tConvectiveVelocity = sqrt(tConvectiveVelocity);

        // calculate stable time step
        tTimeStep(aOrdinal) = tElemSize(aOrdinal) / (tConvectiveVelocity + tArtificialCompressibility(aOrdinal));
    }, "calculate stable time step");

    return tTimeStep;
}

inline void
enforce_boundary_condition
(const Plato::LocalOrdinalVector & aBcDofs,
 const Plato::ScalarVector       & aBcValues,
 Plato::ScalarVector             & aState)
{
    auto tLength = aBcValues.size();
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tLength), LAMBDA_EXPRESSION(const Plato::OrdinalType & aOrdinal)
    {
        auto tDOF = aBcDofs(aOrdinal);
        aState(tDOF) = aBcValues(aOrdinal);
    }, "enforce boundary condition");
}

inline Plato::Scalar
calculate_stopping_criterion
(const Plato::ScalarVector& aTimeStep,
 const Plato::ScalarVector& aCurrentState,
 const Plato::ScalarVector& aPreviousState,
 const Plato::ScalarVector& aArtificialCompressibility)
{
    Plato::Scalar tResidual(0);
    auto tLength = aCurrentState.size();
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tLength), LAMBDA_EXPRESSION(const Plato::OrdinalType & aOrdinal)
    {
        auto tDeltaPressOverTimeStep = aCurrentState(aOrdinal) - aPreviousState(aOrdinal) / aTimeStep(aOrdinal);
        tResidual += ( static_cast<Plato::Scalar>(1) /
            ( aArtificialCompressibility(aOrdinal) * aArtificialCompressibility(aOrdinal) ) )
                * (tDeltaPressOverTimeStep * tDeltaPressOverTimeStep);

    }, "enforce boundary condition");

    return tResidual;
}

inline Plato::Scalar
calculate_explicit_solve_convergence_criterion
(const Plato::Hyperbolic::FluidMechanics::States& aStates)
{
    auto tTimeStep = aStates.getVector("time step");
    auto tCurrentPressure = aStates.getVector("current mass");
    auto tPreviousPressure = aStates.getVector("previous mass");
    auto tArtificialCompress = aStates.getVector("artificial compressibility");

    auto tError = Plato::cbs::calculate_stopping_criterion(tTimeStep, tCurrentPressure, tPreviousPressure, tArtificialCompress);

    return tError;
}

inline Plato::Scalar
calculate_semi_implicit_solve_convergence_criterion
(const Plato::Hyperbolic::FluidMechanics::States& aStates)
{
    std::vector<Plato::Scalar> tErrors;

    // pressure error
    auto tTimeStep = aStates.getVector("time step");
    auto tCurrentState = aStates.getVector("current mass");
    auto tPreviousState = aStates.getVector("previous mass");
    auto tArtificialCompress = aStates.getVector("artificial compressibility");
    auto tMyResidual = Plato::cbs::calculate_stopping_criterion(tTimeStep, tCurrentState, tPreviousState, tArtificialCompress);
    Plato::Scalar tMyMaxStateError(0);
    Plato::blas1::max(tMyResidual, tMyMaxStateError);
    tErrors.push_back(tMyMaxStateError);

    // velocity error
    tCurrentState = aStates.getVector("current momentum");
    tPreviousState = aStates.getVector("previous momentum");
    tMyResidual = Plato::cbs::calculate_stopping_criterion(tTimeStep, tCurrentState, tPreviousState, tArtificialCompress);
    tMyMaxStateError(0.0);
    Plato::blas1::max(tMyResidual, tMyMaxStateError);
    tErrors.push_back(tMyMaxStateError);

    // temperature error
    tCurrentState = aStates.getVector("current energy");
    tPreviousState = aStates.getVector("previous energy");
    tMyResidual = Plato::cbs::calculate_stopping_criterion(tTimeStep, tCurrentState, tPreviousState, tArtificialCompress);
    tMyMaxStateError(0.0);
    Plato::blas1::max(tMyResidual, tMyMaxStateError);
    tErrors.push_back(tMyMaxStateError);

    auto tMax = *std::max_element(tErrors.begin(), tErrors.end());

    return tMax;
}

}
// namespace cbs

namespace Hyperbolic
{

class AbstractProblem
{
public:
    virtual ~AbstractProblem() {}

    virtual void output(const Plato::ScalarVector &aControl, const Plato::Hyperbolic::Solution &aSolution)=0;
    virtual Plato::Hyperbolic::Solution solution(const Plato::ScalarVector& aControl) = 0;
    virtual Plato::Scalar criterionValue(const Plato::ScalarVector & aControl, const std::string& aName) = 0;
    virtual Plato::ScalarVector criterionGradient(const Plato::ScalarVector &aControl, const std::string &aName) = 0;
    virtual Plato::ScalarVector criterionGradientX(const Plato::ScalarVector &aControl, const std::string &aName) = 0;
};

template<typename PhysicsT>
class FluidMechanicsProblem
{
private:
    static constexpr auto mNumSpatialDims         = PhysicsT::mNumSpatialDims;         /*!< number of mass dofs per node */
    static constexpr auto mNumMassDofsPerNode     = PhysicsT::mNumMassDofsPerNode;     /*!< number of mass dofs per node */
    static constexpr auto mNumEnergyDofsPerNode   = PhysicsT::mNumEnergyDofsPerNode;   /*!< number of energy dofs per node */
    static constexpr auto mNumMomentumDofsPerNode = PhysicsT::mNumMomentumDofsPerNode; /*!< number of momentum dofs per node */

    Plato::SpatialModel mSpatialModel; /*!< SpatialModel instance contains the mesh, meshsets, domains, etc. */

    bool mIsExplicitSolve = true;
    bool mIsTransientProblem = false;
    Plato::Scalar mPrandtlNumber = 1.0;
    Plato::Scalar mReynoldsNumber = 1.0;
    Plato::Scalar mCBSsolverTolerance = 1e-5;
    Plato::OrdinalType mNumSteps = 100;

    Plato::ScalarMultiVector mMass;
    Plato::ScalarMultiVector mEnergy;
    Plato::ScalarMultiVector mMomentum;
    Plato::ScalarMultiVector mPredictor;

    using VectorFunctionType = Plato::Hyperbolic::FluidMechanics::VectorFunction<PhysicsT>;
    VectorFunctionType mPressureResidual;
    VectorFunctionType mTemperatureResidual;
    VectorFunctionType mVelocityPredictorResidual;
    VectorFunctionType mVelocityCorrectorResidual;

    using MassT     = typename Plato::MassConservation<PhysicsT::mNumSpatialDims, PhysicsT::mNumControls>;
    using EnergyT   = typename Plato::EnergyConservation<PhysicsT::mNumSpatialDims, PhysicsT::mNumControls>;
    using MomentumT = typename Plato::MomentumConservation<PhysicsT::mNumSpatialDims, PhysicsT::mNumControls>;
    Plato::EssentialBCs<MomentumT> mVelocityStateBoundaryConditions;
    Plato::EssentialBCs<MassT>     mPressureStateBoundaryConditions;
    Plato::EssentialBCs<EnergyT>   mTemperatureStateBoundaryConditions;

    std::shared_ptr<Plato::AbstractSolver> mVectorFieldSolver;
    std::shared_ptr<Plato::AbstractSolver> mScalarFieldSolver;

public:
    FluidMechanicsProblem
    (Omega_h::Mesh &aMesh,
     Omega_h::MeshSets &aMeshSets,
     Teuchos::ParameterList &aProblemParams,
     Comm::Machine aMachine)
    {
        Plato::SolverFactory tSolverFactory(aProblemParams.sublist("Linear Solver"));
        mScalarFieldSolver = tSolverFactory.create(aMesh, aMachine, mNumMassDofsPerNode);
        mVectorFieldSolver = tSolverFactory.create(aMesh, aMachine, mNumMomentumDofsPerNode);
    }

    void output(const Plato::ScalarVector &aControl, const Plato::Hyperbolic::Solution &aSolution)
    {
    }

    Plato::Hyperbolic::Solution solution(const Plato::ScalarVector& aControl)
    {
        Plato::Hyperbolic::FluidMechanics::States tStates;
        tStates.setVector("control", aControl);

        this->calculateElemCharacteristicSize(tStates);
        for (decltype(mNumSteps) tStep = 1; tStep < mNumSteps; tStep++)
        {
            tStates.setScalar("step index", tStep);
            this->setStates(tStates);

            this->calculateStableTimeStep(tStates);
            this->calculateVelocityPredictor(tStates);
            this->calculatePressureState(tStates);
            this->calculateVelocityCorrector(tStates);
            this->calculateTemperatureState(tStates);

            this->enforceVelocityBoundaryConditions(tStates);
            this->enforcePressureBoundaryConditions(tStates);
            this->enforceTemperatureBoundaryConditions(tStates);

            if(this->checkStoppingCriteria(tStates))
            {
                break;
            }
        }

        Plato::Hyperbolic::Solution tSolution;
        tSolution.set("mass state", mMass);
        tSolution.set("energy state", mEnergy);
        tSolution.set("momentum state", mMomentum);
        return tSolution;
    }

    Plato::Scalar criterionValue(const Plato::ScalarVector & aControl, const std::string& aName)
    {
    }

    Plato::ScalarVector criterionGradient(const Plato::ScalarVector &aControl, const std::string &aName)
    {
    }

    Plato::ScalarVector criterionGradientX(const Plato::ScalarVector &aControl, const std::string &aName)
    {
    }

private:
    bool checkStoppingCriteria(const Plato::Hyperbolic::FluidMechanics::States& aStates)
    {
        bool tStop = false;
        if(!mIsTransientProblem)
        {
            Plato::Scalar tCriterionValue(0.0);
            if(mIsExplicitSolve)
            {
                tCriterionValue = Plato::cbs::calculate_explicit_solve_convergence_criterion(aStates);
            }
            else
            {
                tCriterionValue = Plato::cbs::calculate_semi_implicit_solve_convergence_criterion(aStates);
            }

            if(tCriterionValue < mCBSsolverTolerance)
            {
                tStop = true;
            }
        }
        return tStop;
    }
    void calculateElemCharacteristicSize(Plato::Hyperbolic::FluidMechanics::States& aStates)
    {
        Plato::ScalarVector tElementSize;
        Plato::NodeCoordinate<mNumSpatialDims> tNodeCoordinates;
        auto tNumCells = mSpatialModel.Mesh.nverts();
        auto tElemCharacteristicSize =
            Plato::cbs::calculate_element_characteristic_size<mNumSpatialDims>(tNumCells, tNodeCoordinates);
        aStates.setVector("element size", tElemCharacteristicSize);
    }

    void calculateStableTimeStep(Plato::Hyperbolic::FluidMechanics::States& aStates)
    {
        auto tArtificialCompressibility = Plato::cbs::calculate_artificial_compressibility(aStates);
        aStates.setVector("artificial compressibility", tArtificialCompressibility);
        auto tTimeStep = Plato::cbs::calculate_stable_time_step(aStates);
        if(mIsTransientProblem)
        {
            Plato::Scalar tMinTimeStep(0);
            Plato::blas1::min(tTimeStep, tMinTimeStep);
            Plato::blas1::fill(tMinTimeStep, tTimeStep);
            auto tCurrentTimeStepIndex = aStates.getScalar("step index");
            auto tCurrentTime = tMinTimeStep * static_cast<Plato::Scalar>(tCurrentTimeStepIndex);
            aStates.setScalar("current time", tCurrentTime);
        }
        aStates.setVector("time step", tTimeStep);
    }

    void enforceVelocityBoundaryConditions(Plato::Hyperbolic::FluidMechanics::States& aStates)
    {
        Plato::ScalarVector tBcValues;
        Plato::LocalOrdinalVector tBcDofs;
        if(mIsTransientProblem)
        {
            auto tCurrentTime = aStates.getScalar("current time");
            mVelocityStateBoundaryConditions.get(tBcDofs, tBcValues, tCurrentTime);
        }
        else
        {
            mVelocityStateBoundaryConditions.get(tBcDofs, tBcValues);
        }
        auto tCurrentVelocity = aStates.getVector("current momentum");
        Plato::cbs::enforce_boundary_condition(tBcDofs, tBcValues, tCurrentVelocity);
    }

    void enforcePressureBoundaryConditions(Plato::Hyperbolic::FluidMechanics::States& aStates)
    {
        Plato::ScalarVector tBcValues;
        Plato::LocalOrdinalVector tBcDofs;
        if(mIsTransientProblem)
        {
            auto tCurrentTime = aStates.getScalar("current time");
            mPressureStateBoundaryConditions.get(tBcDofs, tBcValues, tCurrentTime);
        }
        else
        {
            mPressureStateBoundaryConditions.get(tBcDofs, tBcValues);
        }
        auto tCurrentPressure = aStates.getVector("current mass");
        Plato::cbs::enforce_boundary_condition(tBcDofs, tBcValues, tCurrentPressure);
    }

    void enforceTemperatureBoundaryConditions(Plato::Hyperbolic::FluidMechanics::States& aStates)
    {
        Plato::ScalarVector tBcValues;
        Plato::LocalOrdinalVector tBcDofs;
        if(mIsTransientProblem)
        {
            auto tCurrentTime = aStates.getScalar("current time");
            mTemperatureStateBoundaryConditions.get(tBcDofs, tBcValues, tCurrentTime);
        }
        else
        {
            mTemperatureStateBoundaryConditions.get(tBcDofs, tBcValues);
        }
        auto tCurrentTemperature = aStates.getVector("current energy");
        Plato::cbs::enforce_boundary_condition(tBcDofs, tBcValues, tCurrentTemperature);
    }

    void setStates(Plato::Hyperbolic::FluidMechanics::States& aStates)
    {
        Plato::OrdinalType tStep = aStates.getScalar("step index");
        auto tPrevStep = tStep - 1;
        auto tMomentumPredictor = Kokkos::subview(mPredictor, tStep, Kokkos::ALL());
        auto tCurrentMass       = Kokkos::subview(mMass,      tStep, Kokkos::ALL());
        auto tCurrentMomentum   = Kokkos::subview(mMomentum,  tStep, Kokkos::ALL());
        auto tCurrentEnergy     = Kokkos::subview(mEnergy,    tStep, Kokkos::ALL());
        aStates.setVector("current mass", tCurrentMass);
        aStates.setVector("current energy", tCurrentEnergy);
        aStates.setVector("current momentum", tCurrentMomentum);
        aStates.setVector("momentum predictor", tMomentumPredictor);

        auto tPreviousMass     = Kokkos::subview(mMass,     tPrevStep, Kokkos::ALL());
        auto tPreviousMomentum = Kokkos::subview(mMomentum, tPrevStep, Kokkos::ALL());
        auto tPreviousEnergy   = Kokkos::subview(mEnergy,   tPrevStep, Kokkos::ALL());
        aStates.setVector("previous mass", tPreviousMass);
        aStates.setVector("previous energy", tPreviousEnergy);
        aStates.setVector("previous momentum", tPreviousMomentum);
    }

    void calculateVelocityCorrector(Plato::Hyperbolic::FluidMechanics::States& aStates)
    {
        // calculate current residual and jacobian matrix
        auto tResidualCorrector = mVelocityCorrectorResidual.value(aStates);
        auto tJacobianCorrector = mVelocityCorrectorResidual.gradient_uc(aStates);

        // solve momentum corrector equation (consistent or mass lumped)
        Plato::ScalarVector tDeltaVelocity("increment", tResidualCorrector.size());
        Plato::blas1::fill(static_cast<Plato::Scalar>(0.0), tDeltaVelocity);
        mScalarFieldSolver->solve(*tJacobianCorrector, tDeltaVelocity, tResidualCorrector);

        // update velocity
        auto tCurrentVelocity = aStates.getVector("current momentum");
        auto tPreviousVelocity = aStates.getVector("previous momentum");
        Plato::blas1::copy(tPreviousVelocity, tCurrentVelocity);
        Plato::blas1::axpy(1.0, tDeltaVelocity, tCurrentVelocity);
    }

    void calculateVelocityPredictor(Plato::Hyperbolic::FluidMechanics::States& aStates)
    {
        // calculate current residual and jacobian matrix
        auto tResidualPredictor = mVelocityPredictorResidual.value(aStates);
        auto tJacobianPredictor = mVelocityPredictorResidual.gradient_du(aStates);

        // solve momentum predictor equation (consistent or mass lumped)
        auto tDeltaVelocityPredictor = aStates.getVector("momentum predictor");
        Plato::blas1::fill(static_cast<Plato::Scalar>(0.0), tDeltaVelocityPredictor);
        mVectorFieldSolver->solve(*tJacobianPredictor, tDeltaVelocityPredictor, tResidualPredictor);
    }

    void calculatePressureState(Plato::Hyperbolic::FluidMechanics::States& aStates)
    {
        // calculate current residual and jacobian matrix
        auto tResidualPressure = mPressureResidual.value(aStates);
        auto tJacobianPressure = mPressureResidual.gradient_mc(aStates);

        // solve mass equation (consistent or mass lumped)
        Plato::ScalarVector tDeltaPressure("increment", tResidualPressure.size());
        Plato::blas1::fill(static_cast<Plato::Scalar>(0.0), tDeltaPressure);
        mScalarFieldSolver->solve(*tJacobianPressure, tDeltaPressure, tResidualPressure);

        // update pressure
        auto tCurrentPressure = aStates.getVector("current mass");
        auto tPreviousPressure = aStates.getVector("previous mass");
        Plato::blas1::copy(tPreviousPressure, tCurrentPressure);
        Plato::blas1::axpy(1.0, tDeltaPressure, tCurrentPressure);
    }

    void calculateTemperatureState(Plato::Hyperbolic::FluidMechanics::States& aStates)
    {
        // calculate current residual and jacobian matrix
        auto tResidualTemperature = mTemperatureResidual.value(aStates);
        auto tJacobianTemperature = mTemperatureResidual.gradient_ec(aStates);

        // solve energy equation (consistent or mass lumped)
        Plato::ScalarVector tDeltaTemperature("increment", tResidualTemperature.size());
        Plato::blas1::fill(static_cast<Plato::Scalar>(0.0), tDeltaTemperature);
        mScalarFieldSolver->solve(*tJacobianTemperature, tDeltaTemperature, tResidualTemperature);

        // update temperature
        auto tCurrentTemperature = aStates.getVector("current energy");
        auto tPreviousTemperature = aStates.getVector("previous energy");
        Plato::blas1::copy(tPreviousTemperature, tCurrentTemperature);
        Plato::blas1::axpy(1.0, tDeltaTemperature, tCurrentTemperature);
    }

    void checkEssentialBoundaryConditions()
    {
        auto tEssentialBoundaryConditionsAreEmpty = mTemperatureStateBoundaryConditions.empty()
            && mPressureStateBoundaryConditions.empty() && mVelocityStateBoundaryConditions.empty();
        if(tEssentialBoundaryConditionsAreEmpty)
        {
            THROWERR("Essential Boundary Conditions are empty.")
        }
    }

    void readTemperatureBoundaryConditions(Teuchos::ParameterList& aInputs)
    {
        Plato::ScalarVector tBcValues;
        Plato::LocalOrdinalVector tBcDofs;
        if(aInputs.isSublist("Temperature Boundary Conditions"))
        {
            auto tVelocityBCs = aInputs.sublist("Temperature Boundary Conditions");
            if(tVelocityBCs.isSublist("Essential Boundary Conditions"))
            {
                auto tParamListBCs = aInputs.sublist("Essential Boundary Conditions");
                mTemperatureStateBoundaryConditions = Plato::EssentialBCs<EnergyT>(tParamListBCs, mSpatialModel.MeshSets);
            }
        }
    }

    void readPressureBoundaryConditions(Teuchos::ParameterList& aInputs)
    {
        Plato::ScalarVector tBcValues;
        Plato::LocalOrdinalVector tBcDofs;
        if(aInputs.isSublist("Pressure Boundary Conditions"))
        {
            auto tVelocityBCs = aInputs.sublist("Pressure Boundary Conditions");
            if(tVelocityBCs.isSublist("Essential Boundary Conditions"))
            {
                auto tParamListBCs = aInputs.sublist("Essential Boundary Conditions");
                mPressureStateBoundaryConditions = Plato::EssentialBCs<MassT>(tParamListBCs, mSpatialModel.MeshSets);
            }
        }
    }

    void readVelocityBoundaryConditions(Teuchos::ParameterList& aInputs)
    {
        Plato::ScalarVector tBcValues;
        Plato::LocalOrdinalVector tBcDofs;
        if(aInputs.isSublist("Velocity Boundary Conditions"))
        {
            auto tVelocityBCs = aInputs.sublist("Velocity Boundary Conditions");
            if(tVelocityBCs.isSublist("Essential Boundary Conditions"))
            {
                auto tParamListBCs = aInputs.sublist("Essential Boundary Conditions");
                mVelocityStateBoundaryConditions = Plato::EssentialBCs<MomentumT>(tParamListBCs, mSpatialModel.MeshSets);
            }
        }
    }
};

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
