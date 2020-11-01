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
#include "BLAS2.hpp"
#include "Simplex.hpp"
#include "NaturalBCs.hpp"
#include "WorksetBase.hpp"
#include "SpatialModel.hpp"
#include "EssentialBCs.hpp"
#include "ProjectToNode.hpp"
#include "PlatoUtilities.hpp"
#include "SimplexFadTypes.hpp"
#include "PlatoMathHelpers.hpp"
#include "PlatoAbstractProblem.hpp"
#include "Plato_TopOptFunctors.hpp"
#include "InterpolateFromNodal.hpp"
#include "LinearTetCubRuleDegreeOne.hpp"

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

inline std::string is_valid_function(const std::string& aInput)
{
    std::vector<std::string> tValidKeys = {"scalar function", "vector function"};
    auto tLowerKey = Plato::tolower(aInput);
    if(std::find(tValidKeys.begin(), tValidKeys.end(), tLowerKey) == tValidKeys.end())
    {
        THROWERR(std::string("Input key with tag '") + tLowerKey + "' is not a valid vector function.")
    }
    return tLowerKey;
}

template <typename T>
inline T parse_dimensionless_property
(const Teuchos::ParameterList & aInputs,
 const std::string            & aTag)
{
    auto tSublist = aInputs.sublist("Dimensionless Properties");
    if(tSublist.isParameter(aTag))
    {
        auto tOutput = tSublist.get<T>(aTag);
        return tOutput;
    }
    else
    {
        THROWERR(std::string("") + aTag + " must be defined for analysis. Use '") + aTag +
            "' keyword inside sublist 'Dimensionless Properties' to define its value")
    }
}

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

// is_fad<TypesT, T>::value is true if T is of any AD type defined TypesT.
//
template <typename TypesT, typename T>
struct is_fad {
  static constexpr bool value = std::is_same< T, typename TypesT::StateFad     >::value ||
                                std::is_same< T, typename TypesT::ControlFad   >::value ||
                                std::is_same< T, typename TypesT::ConfigFad    >::value ||
                                std::is_same< T, typename TypesT::NodeStateFad >::value ||
                                std::is_same< T, typename TypesT::LocalStateFad >::value;
};


// which_fad<TypesT,T1,T2>::type returns:
// -- compile error  if T1 and T2 are both AD types defined in TypesT,
// -- T1             if only T1 is an AD type in TypesT,
// -- T2             if only T2 is an AD type in TypesT,
// -- T2             if neither are AD types.
//
template <typename TypesT, typename T1, typename T2>
struct which_fad {
  static_assert( !(is_fad<TypesT,T1>::value && is_fad<TypesT,T2>::value), "Only one template argument can be an AD type.");
  using type = typename std::conditional< is_fad<TypesT,T1>::value, T1, T2 >::type;
};


// fad_type_t<PhysicsT,T1,T2,T3,...,TN> returns:
// -- compile error  if more than one of T1,...,TN is an AD type in SimplexFadTypes<PhysicsT>,
// -- type TI        if only TI is AD type in SimplexFadTypes<PhysicsT>,
// -- TN             if none of TI are AD type in SimplexFadTypes<PhysicsT>.
//
template <typename TypesT, typename ...P> struct fad_type;
template <typename TypesT, typename T> struct fad_type<TypesT, T> { using type = T; };
template <typename TypesT, typename T, typename ...P> struct fad_type<TypesT, T, P ...> {
  using type = typename which_fad<TypesT, T, typename fad_type<TypesT, P...>::type>::type;
};
template <typename PhysicsT, typename ...P> using fad_type_t = typename fad_type<SimplexFadTypes<PhysicsT>,P...>::type;

struct States
{
private:
    std::string mType;
    std::unordered_map<std::string, Plato::Scalar> mScalars;
    std::unordered_map<std::string, Plato::ScalarVector> mStates;

public:
    decltype(mType) function() const
    {
        return mType;
    }

    void function(const decltype(mType)& aInput)
    {
        mType = aInput;
    }

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

    bool isVectorMapEmpty() const
    {
        return mStates.empty();
    }

    bool isScalarMapEmpty() const
    {
        return mStates.empty();
    }
};
typedef States Dual;

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
struct ResultTypes : EvaluationTypes<SimplexPhysicsT>
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
    using MomentumCorrectorScalarType = Plato::Scalar;
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
  using MomentumCorrectorScalarType = Plato::Scalar;
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
  using MomentumCorrectorScalarType = Plato::Scalar;
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
  using MomentumCorrectorScalarType = Plato::Scalar;
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
  using MomentumCorrectorScalarType = Plato::Scalar;
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
  using MomentumCorrectorScalarType = Plato::Scalar;
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
  using MomentumCorrectorScalarType = Plato::Scalar;
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
  using MomentumCorrectorScalarType = Plato::Scalar;
};

template <typename SimplexPhysicsT>
struct GradMomentumCorrectorTypes : EvaluationTypes<SimplexPhysicsT>
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
  using MomentumPredictorScalarType = Plato::Scalar;
  using MomentumCorrectorScalarType = FadType;
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
  using MomentumCorrectorScalarType = Plato::Scalar;
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
  using MomentumCorrectorScalarType = Plato::Scalar;
};

template <typename SimplexPhysicsT>
struct Evaluation
{
   using Residual         = ResultTypes<SimplexPhysicsT>;
   using GradConfig       = GradControlTypes<SimplexPhysicsT>;
   using GradControl      = GradConfigTypes<SimplexPhysicsT>;
   using GradMassCurr     = GradCurrentMassTypes<SimplexPhysicsT>;
   using GradMassPrev     = GradCurrentMassTypes<SimplexPhysicsT>;
   using GradEnergyCurr   = GradCurrentEnergyTypes<SimplexPhysicsT>;
   using GradEnergyPrev   = GradPreviousEnergyTypes<SimplexPhysicsT>;
   using GradMomentumCurr = GradPreviousMomentumTypes<SimplexPhysicsT>;
   using GradMomentumPrev = GradPreviousMomentumTypes<SimplexPhysicsT>;
   using GradPredictor    = GradMomentumPredictorTypes<SimplexPhysicsT>;
   using GradCorrector    = GradMomentumCorrectorTypes<SimplexPhysicsT>;
};


template<typename PhysicsT, typename EvaluationT>
struct WorkSets
{
private:
    static constexpr auto mNumControls         = PhysicsT::SimplexT::mNumControl;             /*!< number of design variable fields */
    static constexpr auto mNumSpatialDims      = PhysicsT::SimplexT::mNumSpatialDims;         /*!< number of spatial dimensions */
    static constexpr auto mNumNodesPerCell     = PhysicsT::SimplexT::mNumNodesPerCell;        /*!< number of nodes per cell */
    static constexpr auto mNumPressDofsPerCell = PhysicsT::SimplexT::mNumMassDofsPerCell;     /*!< number of mass dofs per cell */
    static constexpr auto mNumTempDofsPerCell  = PhysicsT::SimplexT::mNumEnergyDofsPerCell;   /*!< number of energy dofs per cell */
    static constexpr auto mNumVelDofsPerCell   = PhysicsT::SimplexT::mNumMomentumDofsPerCell; /*!< number of momentum dofs per cell */

    Plato::OrdinalType mNumCells;

    Plato::ScalarArray3DT<typename EvaluationT::ConfigScalarType> mConfiguration;

    Plato::ScalarMultiVectorT<Plato::Scalar> mTimeStep;
    Plato::ScalarMultiVectorT<typename EvaluationT::ControlScalarType> mControls;
    Plato::ScalarMultiVectorT<typename EvaluationT::CurrentMassScalarType> mCurrentPress;
    Plato::ScalarMultiVectorT<typename EvaluationT::CurrentEnergyScalarType> mCurrentTemp;
    Plato::ScalarMultiVectorT<typename EvaluationT::CurrentMomentumScalarType> mCurrentVel;
    Plato::ScalarMultiVectorT<typename EvaluationT::PreviousMassScalarType> mPreviousPress;
    Plato::ScalarMultiVectorT<typename EvaluationT::PreviousEnergyScalarType> mPreviousTemp;
    Plato::ScalarMultiVectorT<typename EvaluationT::PreviousMomentumScalarType> mPreviousVel;
    Plato::ScalarMultiVectorT<typename EvaluationT::MomentumPredictorScalarType> mMomentumPredictor;
    Plato::ScalarMultiVectorT<typename EvaluationT::MomentumCorrectorScalarType> mMomentumCorrector;

public:
    explicit WorkSets(const Plato::OrdinalType& aNumCells) :
        mNumCells(aNumCells),
        mConfiguration("Configuration Workset", aNumCells, mNumNodesPerCell, mNumSpatialDims),
        mTimeStep("Time Step Workset", aNumCells, mNumNodesPerCell),
        mControls("Control Workset", aNumCells, mNumNodesPerCell),
        mCurrentPress("Current Mass Workset", aNumCells, mNumPressDofsPerCell),
        mCurrentTemp("Current Energy Workset", aNumCells, mNumTempDofsPerCell),
        mCurrentVel("Current Momentum Workset", aNumCells, mNumVelDofsPerCell),
        mPreviousPress("Previous Mass Workset", aNumCells, mNumPressDofsPerCell),
        mPreviousTemp("Previous Energy Workset", aNumCells, mNumTempDofsPerCell),
        mPreviousVel("Previous Momentum Workset", aNumCells, mNumVelDofsPerCell),
        mMomentumPredictor("Momentum Predictor Workset", aNumCells, mNumVelDofsPerCell),
        mMomentumCorrector("Momentum Corrector Workset", aNumCells, mNumVelDofsPerCell)
    {}

    decltype(mNumCells) numCells() const
    {
        return mNumCells;
    }

    decltype(mTimeStep) timeStep()
    {
        return mTimeStep;
    }

    decltype(mControls) control()
    {
        return mControls;
    }

    decltype(mConfiguration) configuration()
    {
        return mConfiguration;
    }

    decltype(mCurrentPress) currentPressure()
    {
        return mCurrentPress;
    }

    decltype(mCurrentTemp) currentTemperature()
    {
        return mCurrentTemp;
    }

    decltype(mCurrentVel) currentVelocity()
    {
        return mCurrentVel;
    }

    decltype(mPreviousPress) previousPressure()
    {
        return mPreviousPress;
    }

    decltype(mPreviousTemp)& previousTemperature()
    {
        return mPreviousTemp;
    }

    decltype(mPreviousVel) previousVelocity()
    {
        return mPreviousVel;
    }

    decltype(mMomentumPredictor) predictor()
    {
        return mMomentumPredictor;
    }

    decltype(mMomentumCorrector) corrector()
    {
        return mMomentumCorrector;
    }
};

template<typename PhysicsT, typename EvaluationT>
class AbstractScalarFunction
{
public:
    AbstractScalarFunction(){}
    virtual ~AbstractScalarFunction(){}

    virtual void evaluate
    (const Plato::Hyperbolic::FluidMechanics::WorkSets<typename PhysicsT, typename EvaluationT> & aWorkSets,
     Plato::ScalarVectorT<typename EvaluationT::ResultScalarType>                         & aResult) const = 0;
};
// class AbstractScalarFunction

/******************************************************************************/
/*! scalar function class

   This class takes as a template argument a scalar function in the form:

   \f$ J = J(\phi, U^k, P^k, T^k, X) \f$

   and manages the evaluation of the function and derivatives with respect to
   control, \f$\phi\f$, momentum state, \f$ U^k \f$, mass state, \f$ P^k \f$,
   energy state, \f$ T^k \f$, and configuration, \f$ X \f$.

*/
/******************************************************************************/
class CriterionBase
{
private:
    using PrimalStates = Plato::Hyperbolic::FluidMechanics::States;

public:
    virtual ~CriterionBase(){}

    /******************************************************************************//**
     * \brief Return function name
     * \return user defined function name
     **********************************************************************************/
    virtual std::string name() const = 0;

    virtual Plato::Scalar value
    (const Plato::ScalarVector & aControls,
     const PrimalStates & aStates) const = 0;

    virtual Plato::ScalarVector gradientConfig
    (const Plato::ScalarVector & aControls,
     const PrimalStates & aStates) const = 0;

    virtual Plato::ScalarVector gradientControl
    (const Plato::ScalarVector & aControls,
     const PrimalStates & aStates) const = 0;

    virtual Plato::ScalarVector gradientCurrentPress
    (const Plato::ScalarVector & aControls,
     const PrimalStates & aStates) const = 0;

    virtual Plato::ScalarVector gradientCurrentTemp
    (const Plato::ScalarVector & aControls,
     const PrimalStates & aStates) const = 0;

    virtual Plato::ScalarVector gradientCurrentVel
    (const Plato::ScalarVector & aControls,
     const PrimalStates & aStates) const = 0;
};
// class ScalarFunctionBase

template<typename PhysicsT>
class PhysicsScalarFunction : public Plato::Hyperbolic::FluidMechanics::CriterionBase
{
private:
    std::string mScalarFuncName;

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
    using Result           = typename Plato::Hyperbolic::FluidMechanics::Evaluation<typename PhysicsT::SimplexT>::Residual;
    using GradConfig       = typename Plato::Hyperbolic::FluidMechanics::Evaluation<typename PhysicsT::SimplexT>::GradConfig;
    using GradControl      = typename Plato::Hyperbolic::FluidMechanics::Evaluation<typename PhysicsT::SimplexT>::GradControl;
    using GradCurrMass     = typename Plato::Hyperbolic::FluidMechanics::Evaluation<typename PhysicsT::SimplexT>::GradMassCurr;
    using GradCurrEnergy   = typename Plato::Hyperbolic::FluidMechanics::Evaluation<typename PhysicsT::SimplexT>::GradEnergyCurr;
    using GradCurrMomentum = typename Plato::Hyperbolic::FluidMechanics::Evaluation<typename PhysicsT::SimplexT>::GradMomentumCurr;

    // element residual functions
    using ResultFunc           = std::shared_ptr<Plato::Hyperbolic::FluidMechanics::AbstractScalarFunction<typename PhysicsT::SimplexT, Result>>;
    using GradConfigFunc       = std::shared_ptr<Plato::Hyperbolic::FluidMechanics::AbstractScalarFunction<typename PhysicsT::SimplexT, GradConfig>>;
    using GradControlFunc      = std::shared_ptr<Plato::Hyperbolic::FluidMechanics::AbstractScalarFunction<typename PhysicsT::SimplexT, GradControl>>;
    using GradCurrMassFunc     = std::shared_ptr<Plato::Hyperbolic::FluidMechanics::AbstractScalarFunction<typename PhysicsT::SimplexT, GradCurrMass>>;
    using GradCurrEnergyFunc   = std::shared_ptr<Plato::Hyperbolic::FluidMechanics::AbstractScalarFunction<typename PhysicsT::SimplexT, GradCurrEnergy>>;
    using GradCurrMomentumFunc = std::shared_ptr<Plato::Hyperbolic::FluidMechanics::AbstractScalarFunction<typename PhysicsT::SimplexT, GradCurrMomentum>>;

    // element scalar functions per element block, i.e. domain
    std::unordered_map<std::string, ResultFunc>           mValueFuncs;
    std::unordered_map<std::string, GradConfigFunc>       mGradConfigFuncs;
    std::unordered_map<std::string, GradControlFunc>      mGradControlFuncs;
    std::unordered_map<std::string, GradCurrMassFunc>     mGradCurrMassFuncs;
    std::unordered_map<std::string, GradCurrEnergyFunc>   mGradCurrEnergyFuncs;
    std::unordered_map<std::string, GradCurrMomentumFunc> mGradCurrMomentumFuncs;

    Plato::DataMap& mDataMap;
    const Plato::SpatialModel& mSpatialModel;
    Plato::NodeCoordinate<mNumSpatialDims> mNodeCoordinate;  /*!< node coordinates metadata */

    // local-to-global physics degrees of freedom maps
    Plato::VectorEntryOrdinal<mNumSpatialDims, mNumSpatialDims>         mConfigEntryOrdinal;        /*!< configuration local-to-global ID map */
    Plato::VectorEntryOrdinal<mNumSpatialDims, mNumControlPerNode>      mControlEntryOrdinal;       /*!< control local-to-global ID map */
    Plato::VectorEntryOrdinal<mNumSpatialDims, mNumMassDofsPerCell>     mMassStateEntryOrdinal;     /*!< mass state local-to-global ID map */
    Plato::VectorEntryOrdinal<mNumSpatialDims, mNumEnergyDofsPerCell>   mEnergyStateEntryOrdinal;   /*!< energy state local-to-global ID map */
    Plato::VectorEntryOrdinal<mNumSpatialDims, mNumMomentumDofsPerCell> mMomentumStateEntryOrdinal; /*!< momentum state local-to-global ID map */

    using PrimalStates = Plato::Hyperbolic::FluidMechanics::States;

public:
    PhysicsScalarFunction
    (const Plato::SpatialModel & aSpatialModel,
     Plato::DataMap & aDataMap,
     Teuchos::ParameterList & aInputs,
     std::string & aName):
        mSpatialModel   (aSpatialModel),
        mDataMap        (aDataMap),
        mScalarFuncName (aName)
    {
        this->initialize(aInputs);
    }

    std::string name() const
    {
        return mScalarFuncName;
    }

    Plato::Scalar value
    (const Plato::ScalarVector & aControls,
     const PrimalStates & aStates) const
    {
        Result tReturnValue(0.0);
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            auto tName     = tDomain.getDomainName();

            Plato::Hyperbolic::FluidMechanics::WorkSets<PhysicsT, Result> tWorkSets(tNumCells);
            this->setWorkSets(tDomain, aControls, aStates, tWorkSets);

            auto tName = tDomain.getDomainName();
            Plato::ScalarVectorT<Result> tResultWS("cells value", tNumCells);
            mValueFuncs.at(tName)->evaluate(tWorkSets, tResultWS);

            tReturnValue += Plato::local_result_sum<Plato::Scalar>(tNumCells, tResultWS);
        }

        return tReturnValue;
    }

    Plato::ScalarVector gradientConfig
    (const Plato::ScalarVector & aControls,
     const PrimalStates & aStates) const
    {
        const auto tNumNodes = mSpatialModel.Mesh.nverts();
        Plato::ScalarVector tGradient("gradient wrt configuration", mNumSpatialDims * tNumNodes);
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            auto tName     = tDomain.getDomainName();

            Plato::Hyperbolic::FluidMechanics::WorkSets<PhysicsT, GradConfig> tWorkSets(tNumCells);
            this->setWorkSets(tDomain, aControls, aStates, tWorkSets);

            auto tName = tDomain.getDomainName();
            Plato::ScalarVectorT<GradConfig> tResultWS("cells value", tNumCells);
            mValueFuncs.at(tName)->evaluate(tWorkSets, tResultWS);

            Plato::assemble_vector_gradient_fad<mNumNodesPerCell, mNumSpatialDims>
                (tDomain, mConfigEntryOrdinal, tResultWS, tGradient);
        }

        return tGradient;
    }

    Plato::ScalarVector gradientControl
    (const Plato::ScalarVector & aControls,
     const PrimalStates & aStates) const
    {
        const auto tNumNodes = mSpatialModel.Mesh.nverts();
        Plato::ScalarVector tGradient("gradient wrt control", mNumControlPerNode * tNumNodes);
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            auto tName     = tDomain.getDomainName();

            Plato::Hyperbolic::FluidMechanics::WorkSets<PhysicsT, GradControl> tWorkSets(tNumCells);
            this->setWorkSets(tDomain, aControls, aStates, tWorkSets);

            auto tName = tDomain.getDomainName();
            Plato::ScalarVectorT<GradControl> tResultWS("cells value", tNumCells);
            mValueFuncs.at(tName)->evaluate(tWorkSets, tResultWS);

            Plato::assemble_vector_gradient_fad<mNumNodesPerCell, mNumControlPerNode>
                (tDomain, mControlEntryOrdinal, tResultWS, tGradient);
        }

        return tGradient;
    }

    Plato::ScalarVector gradientCurrentPress
    (const Plato::ScalarVector & aControls,
     const PrimalStates & aStates) const
    {
        const auto tNumNodes = mSpatialModel.Mesh.nverts();
        Plato::ScalarVector tGradient("gradient wrt current pressure state", mNumMassDofsPerNode * tNumNodes);
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            auto tName     = tDomain.getDomainName();

            Plato::Hyperbolic::FluidMechanics::WorkSets<PhysicsT, GradCurrMass> tWorkSets(tNumCells);
            this->setWorkSets(tDomain, aControls, aStates, tWorkSets);

            auto tName = tDomain.getDomainName();
            Plato::ScalarVectorT<GradCurrMass> tResultWS("cells value", tNumCells);
            mValueFuncs.at(tName)->evaluate(tWorkSets, tResultWS);

            Plato::assemble_vector_gradient_fad<mNumNodesPerCell, mNumMassDofsPerNode>
                (tDomain, mMassStateEntryOrdinal, tResultWS, tGradient);
        }

        return tGradient;
    }

    Plato::ScalarVector gradientCurrentTemp
    (const Plato::ScalarVector & aControls,
     const PrimalStates & aStates) const
    {
        const auto tNumNodes = mSpatialModel.Mesh.nverts();
        Plato::ScalarVector tGradient("gradient wrt current temperature state", mNumEnergyDofsPerNode * tNumNodes);
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            auto tName     = tDomain.getDomainName();

            Plato::Hyperbolic::FluidMechanics::WorkSets<PhysicsT, GradCurrEnergy> tWorkSets(tNumCells);
            this->setWorkSets(tDomain, aControls, aStates, tWorkSets);

            auto tName = tDomain.getDomainName();
            Plato::ScalarVectorT<GradCurrEnergy> tResultWS("cells value", tNumCells);
            mValueFuncs.at(tName)->evaluate(tWorkSets, tResultWS);

            Plato::assemble_vector_gradient_fad<mNumNodesPerCell, mNumEnergyDofsPerNode>
                (tDomain, mEnergyStateEntryOrdinal, tResultWS, tGradient);
        }

        return tGradient;
    }

    Plato::ScalarVector gradientCurrentVel
    (const Plato::ScalarVector & aControls,
     const PrimalStates & aStates) const
    {
        const auto tNumNodes = mSpatialModel.Mesh.nverts();
        Plato::ScalarVector tGradient("gradient wrt current velocity state", mNumMomentumDofsPerNode * tNumNodes);
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            auto tName     = tDomain.getDomainName();

            Plato::Hyperbolic::FluidMechanics::WorkSets<PhysicsT, GradCurrMomentum> tWorkSets(tNumCells);
            this->setWorkSets(tDomain, aControls, aStates, tWorkSets);

            auto tName = tDomain.getDomainName();
            Plato::ScalarVectorT<GradCurrMomentum> tResultWS("cells value", tNumCells);
            mValueFuncs.at(tName)->evaluate(tWorkSets, tResultWS);

            Plato::assemble_vector_gradient_fad<mNumNodesPerCell, mNumMomentumDofsPerNode>
                (tDomain, mMomentumStateEntryOrdinal, tResultWS, tGradient);
        }

        return tGradient;
    }

private:
    void initialize(Teuchos::ParameterList & aInputs)
    {
        typename PhysicsT::FunctionFactory tScalarFuncFactory;

        auto tInputs = aInputs.sublist("Criteria").sublist(mScalarFuncName);
        auto tFunctionType = tInputs.get<std::string>("Scalar Function Type", "not defined");

        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tName = tDomain.getDomainName();

            mValueFuncs[tName] =
                tScalarFuncFactory.template createScalarFunction<Result>
                    (tDomain, mDataMap, aInputs, tFunctionType, mScalarFuncName);

            mGradConfigFuncs[tName] =
                tScalarFuncFactory.template createScalarFunction<GradConfig>
                    (tDomain, mDataMap, aInputs, tFunctionType, mScalarFuncName);

            mGradControlFuncs[tName] =
                tScalarFuncFactory.template createScalarFunction<GradControl>
                    (tDomain, mDataMap, aInputs, tFunctionType, mScalarFuncName);

            mGradCurrMassFuncs[tName] =
                tScalarFuncFactory.template createScalarFunction<GradCurrMass>
                    (tDomain, mDataMap, aInputs, tFunctionType, mScalarFuncName);

            mGradCurrEnergyFuncs[tName] =
                tScalarFuncFactory.template createScalarFunction<GradCurrEnergy>
                    (tDomain, mDataMap, aInputs, tFunctionType, mScalarFuncName);

            mGradCurrMomentumFuncs[tName] =
                tScalarFuncFactory.template createScalarFunction<GradCurrMomentum>
                    (tDomain, mDataMap, aInputs, tFunctionType, mScalarFuncName);
        }
    }

    template<typename WorkSetT>
    void setWorkSets
    (const Plato::SpatialDomain & aDomain,
     const Plato::ScalarVector & aControls,
     const PrimalStates & aState,
     WorkSetT & aWorkSets)
    {
        Plato::workset_state_scalar_scalar<mNumMomentumDofsPerNode, mNumNodesPerCell>
            (aDomain, mMomentumStateEntryOrdinal, aState.getVector("current velocity"), aWorkSets.currentVelocity());

        Plato::workset_state_scalar_scalar<mNumMassDofsPerNode, mNumNodesPerCell>
            (aDomain, mMassStateEntryOrdinal, aState.getVector("current pressure"), aWorkSets.currentPressure());

        Plato::workset_state_scalar_scalar<mNumEnergyDofsPerNode, mNumNodesPerCell>
            (aDomain, mEnergyStateEntryOrdinal, aState.getVector("current temperature"), aWorkSets.currentTemperature());

        Plato::workset_control_scalar_scalar<mNumNodesPerCell>
            (aDomain, mControlEntryOrdinal, aControls, aWorkSets.control());

        Plato::workset_config_scalar<mNumSpatialDims, mNumNodesPerCell>
            (aDomain, mNodeCoordinate, aWorkSets.configuration());
    }
};
// class PhysicsScalarFunction


template<typename PhysicsT, typename EvaluationT>
class AbstractVectorFunctionBase
{
public:
    AbstractVectorFunctionBase(){}
    virtual ~AbstractVectorFunctionBase(){}

    virtual void evaluate
    (const Plato::Hyperbolic::FluidMechanics::WorkSets<typename PhysicsT, typename EvaluationT> & aWorkSets,
     Plato::ScalarMultiVectorT<typename EvaluationT::ResultScalarType>                          & aResult) const = 0;

    virtual void evaluateBoundary
    (const Plato::Hyperbolic::FluidMechanics::WorkSets<typename PhysicsT, typename EvaluationT> & aWorkSets,
     Plato::ScalarMultiVectorT<typename EvaluationT::ResultScalarType>                          & aResult) const = 0;
};
// class AbstractVectorFunction



template<typename PhysicsT, typename EvaluationT>
class VelocityPredictorResidual :
    public Plato::Hyperbolic::FluidMechanics::AbstractVectorFunctionBase<PhysicsT, EvaluationT>
{
private:
    static constexpr auto mNumSpatialDims       = PhysicsT::SimplexT::mNumSpatialDims;         /*!< number of spatial dimensions */
    static constexpr auto mNumNodesPerCell      = PhysicsT::SimplexT::mNumNodesPerCell;        /*!< number of nodes per cell */
    static constexpr auto mNumVelDofsPerCell    = PhysicsT::SimplexT::mNumMomentumDofsPerCell; /*!< number of momentum dofs per cell */
    static constexpr auto mNumTempDofsPerCell   = PhysicsT::SimplexT::mNumEnergyDofsPerCell;   /*!< number of energy dofs per cell */
    static constexpr auto mNumVelDofsPerNode    = PhysicsT::SimplexT::mNumMomentumDofsPerNode; /*!< number of momentum dofs per node */
    static constexpr auto mNumTempDofsPerNode   = PhysicsT::SimplexT::mNumEnergyDofsPerNode;   /*!< number of energy dofs per node */
    static constexpr auto mNumConfigDofsPerCell = mNumSpatialDims * mNumNodesPerCell;          /*!< number of configuration degrees of freedom per cell */

    using ResultT    = typename EvaluationT::ResultScalarType;
    using ConfigT    = typename EvaluationT::ConfigScalarType;
    using ControlT   = typename EvaluationT::ControlScalarType;
    using PrevVelT   = typename EvaluationT::PreviousMomentumScalarType;
    using PrevTempT  = typename EvaluationT::PreviousEnergyScalarType;
    using PredictorT = typename EvaluationT::MomentumPredictorScalarType;

    Plato::DataMap& mDataMap;                   /*!< output database */
    const Plato::SpatialDomain& mSpatialDomain; /*!< Plato spatial model */

    Plato::LinearTetCubRuleDegreeOne<mNumSpatialDims>      mCubatureRule;
    std::shared_ptr<Plato::NaturalBCs<mNumSpatialDims, mNumVelDofsPerNode>> mNeumannLoads; /*!< Neumann loads interface */

    using StateWorkSets = Plato::Hyperbolic::FluidMechanics::WorkSets<PhysicsT, EvaluationT>;

    Plato::Scalar mDaNum = 1.0;
    Plato::Scalar mPrNum = 1.0;
    Plato::Scalar mGrNumExponent = 3.0;
    Plato::Scalar mPrNumConvexityParam = 0.5;
    Plato::Scalar mBrinkmanConvexityParam = 0.5;

    Plato::ScalarVector mGrNum;

public:
    VelocityPredictorResidual
    (const Plato::SpatialDomain & aDomain,
     Plato::DataMap             & aDataMap,
     Teuchos::ParameterList     & aInputs) :
         mDataMap(aDataMap),
         mSpatialDomain(aDomain),
         mCubatureRule(Plato::LinearTetCubRuleDegreeOne<mNumSpatialDims>())
    {
        if(aInputs.isSublist("Traction Boundary Conditions"))
        {
            auto tSublist = aInputs.sublist("Traction Boundary Conditions");
            mNeumannLoads = std::make_shared<Plato::NaturalBCs<mNumSpatialDims, mNumVelDofsPerNode>>(tSublist);
        }
        this->setPenaltyModel(aInputs);
        this->setDimensionlessProperties(aInputs);
    }

    virtual ~VelocityPredictorResidual(){}

    void evaluate
    (const StateWorkSets                & aWorkSets,
     Plato::ScalarMultiVectorT<ResultT> & aResult) const
    {
        using StrainT =
            typename Plato::Hyperbolic::FluidMechanics::fad_type_t<typename PhysicsT::SimplexT, PrevVelT, ConfigT>;

        auto tNumCells = mSpatialDomain.numCells();
        Plato::ScalarVectorT<ConfigT>   tCellVolume("cell weight", tNumCells);
        Plato::ScalarVectorT<PrevTempT> tPrevTempGP("previous temperature at Gauss point", tNumCells);

        Plato::ScalarArray3DT<ConfigT> tGradient("cell gradient", tNumCells, mNumNodesPerCell, mNumSpatialDims);
        Plato::ScalarArray3DT<StrainT> tStrainRate("cell strain rate", tNumCells, mNumSpatialDims, mNumSpatialDims);

        Plato::ScalarMultiVectorT<ResultT>    tStabForce("stabilized force", tNumCells, mNumSpatialDims);
        Plato::ScalarMultiVectorT<PrevVelT>   tPrevVelGP("previous velocity at Gauss point", tNumCells, mNumSpatialDims);
        Plato::ScalarMultiVectorT<PrevVelT>   tVelGrad("velocity gradient", tNumCells, mNumSpatialDims);
        Plato::ScalarMultiVectorT<PredictorT> tPredictorGP("predictor at Gauss point", tNumCells, mNumSpatialDims);

        // set local functors
        Plato::ComputeGradientWorkset<mNumSpatialDims> tComputeGradient;
        Plato::ProjectToNode<mNumSpatialDims, mNumVelDofsPerNode> tCalculateInertialForce;
        Plato::InterpolateFromNodal<mNumSpatialDims, mNumVelDofsPerNode, 0/*offset*/, mNumSpatialDims> tIntrplVectorField;
        Plato::InterpolateFromNodal<mNumSpatialDims, mNumTempDofsPerNode, 0/*offset*/, mNumSpatialDims> tIntrplScalarField;

        // set input state worksets
        auto tConfigWS    = aWorkSets.configuration();
        auto tControlWS   = aWorkSets.control();
        auto tTimeStepWS  = aWorkSets.timeStep();
        auto tPrevVelWS   = aWorkSets.previousVelocity();
        auto tPrevTempWS  = aWorkSets.previousTemperature();
        auto tPredictorWS = aWorkSets.predictor();

        // transfer member data to device
        auto tDaNum = mDaNum;
        auto tPrNum = mPrNum;
        auto tGrNum = mGrNum;
        auto tPowerPenaltySIMP = mGrNumExponent;
        auto tPrNumConvexityParam = mPrNumConvexityParam;
        auto tBrinkmanConvexityParam = mBrinkmanConvexityParam;

        auto tCubWeight = mCubatureRule->getCubWeight();
        auto tBasisFunctions = mCubatureRule->getBasisFunctions();
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
        {
            tComputeGradient(aCellOrdinal, tGradient, tConfigWS, tCellVolume);
            tCellVolume(aCellOrdinal) *= tCubWeight;

            // calculate convective inertial force integral, which are defined as
            // \int_{\Omega_e} N_u^a \left( \frac{\partial}{\partial x_j}(u^h_j u^h_i) \right) d\Omega_e
            tIntrplVectorField(aCellOrdinal, tBasisFunctions, tPrevVelWS, tPrevVelGP);
            for(Plato::OrdinalType tNode = 0; tNode < mNumNodesPerCell; tNode++)
            {
                for(Plato::OrdinalType tDimI = 0; tDimI < mNumSpatialDims; tDimI++)
                {
                    auto tDofIndex = (mNumSpatialDims * tNode) + tDimI;
                    for(Plato::OrdinalType tDimJ = 0; tDimJ < mNumSpatialDims; tDimJ++)
                    {
                        aResult(aCellOrdinal, tDofIndex) += tCellVolume(aCellOrdinal) * tBasisFunctions(tNode) *
                            ( tGradient(aCellOrdinal, tNode, tDimJ) * ( tPrevVelGP(tDimJ) * tPrevVelGP(tDimI) ) );
                        tStabForce(aCellOrdinal, tDimI) += tGradient(aCellOrdinal, tNode, tDimJ) *
                            ( tPrevVelGP(tDimJ) * tPrevVelGP(tDimI) )
                    }
                }
            }

            // calculate strain rate for incompressible flows, which is defined as
            // \frac{1}{2}\left( \frac{\partial u_i}{\partial x_j} + \frac{\partial u_j}{\partial x_i} \right)
            for(Plato::OrdinalType tNode = 0; tNode < mNumNodesPerCell; tNode++)
            {
                for(Plato::OrdinalType tDimI = 0; tDimI < mNumSpatialDims; tDimI++)
                {
                    for(Plato::OrdinalType tDimJ = 0; tDimJ < mNumSpatialDims; tDimJ++)
                    {
                        tStrainRate(aCellOrdinal, tDimI, tDimJ) += static_cast<Plato::Scalar>(0.5) *
                            ( ( tGradient(aCellOrdinal, tNode, tDimJ) * tPrevVelWS(aCellOrdinal, tDimI) )
                            + ( tGradient(aCellOrdinal, tNode, tDimI) * tPrevVelWS(aCellOrdinal, tDimJ) ) );
                    }
                }
            }

            // calculate penalized prandtl number
            ControlT tDensity = Plato::cell_density<mNumNodesPerCell>(aCellOrdinal, tControlWS);
            ControlT tPenalizedPrandtlNum =
                ( tDensity * ( tPrNum * (1.0 - tPrNumConvexityParam) - 1.0 ) + 1.0 ) /
                ( tPrNum * (1.0 + tPrNumConvexityParam * tDensity) );

            // calculate viscous force integral, which are defined as,
            // \int_{\Omega_e}\frac{\partial N_u^a}{\partial x_j}\tau^h_{ij} d\Omega_e
            for(Plato::OrdinalType tNode = 0; tNode < mNumNodesPerCell; tNode++)
            {
                for(Plato::OrdinalType tDimI = 0; tDimI < mNumSpatialDims; tDimI++)
                {
                    auto tDofIndex = (mNumSpatialDims * tNode) + tDimI;
                    for(Plato::OrdinalType tDimJ = 0; tDimJ < mNumSpatialDims; tDimJ++)
                    {
                        aResult(aCellOrdinal, tDofIndex) += tCellVolume(aCellOrdinal) * tGradient(aCellOrdinal, tNode, tDimJ)
                            * ( static_cast<Plato::Scalar>(2.0) * tPenalizedPrandtlNum * tStrainRate(aCellOrdinal, tDimI, tDimJ) );
                    }
                }
            }

            // calculate natural convective force integral, which are defined as
            // \int_{\Omega_e} N_u^a \left(Gr_i Pr^2 T^h \right) d\Omega_e,
            // where e_i is the unit vector in the gravitational direction
            tIntrplScalarField(aCellOrdinal, tBasisFunctions, tPrevTempWS, tPrevTempGP)
            ControlT tPenalizedPrNumSquared = pow(tDensity, tPowerPenaltySIMP) * tPrNum * tPrNum;
            for(Plato::OrdinalType tNode = 0; tNode < mNumNodesPerCell; tNode++)
            {
                for(Plato::OrdinalType tDim = 0; tDim < mNumSpatialDims; tDim++)
                {
                    auto tDofIndex = (mNumSpatialDims * tNode) + tDim;
                    aResult(aCellOrdinal, tDofIndex) += tCellVolume(aCellOrdinal) * tBasisFunctions(tNode)
                        * (tGrNum(tDim) * tPenalizedPrNumSquared * tPrevTempGP);
                    tStabForce(aCellOrdinal, tDim) += tGrNum(tDim) * tPenalizedPrNumSquared * tPrevTempGP;
                }
            }

            // calculate brinkman force integral, which are defined as
            // \int_{\Omega_e} N_u^a (\frac{Pr}{Da} u^h_i) d\Omega
            ControlT tPenalizedBrinkmanCoeff = (tPrNum / tDaNum) * (1.0 - tDensity) / (1.0 + (tBrinkmanConvexityParam * tDensity));
            for(Plato::OrdinalType tNode = 0; tNode < mNumNodesPerCell; tNode++)
            {
                for(Plato::OrdinalType tDim = 0; tDim < mNumSpatialDims; tDim++)
                {
                    auto tDofIndex = (mNumSpatialDims * tNode) + tDim;
                    aResult(aCellOrdinal, tDofIndex) += tCellVolume(aCellOrdinal) *
                        tBasisFunctions(tNode) * (tPenalizedBrinkmanCoeff * tPrevVelGP(tDim));
                    tStabForce(aCellOrdinal, tDim) += tPenalizedBrinkmanCoeff * tPrevVelGP(tDim);
                }
            }

            // calculate stabilizing force integral, which are defined as
            // \int_{\Omega_e} \left( \frac{\partial N_u^a}{\partial x_k} u^h_k \right) F_i^{stab} d\Omega_e
            // where the stabilizing force, F_i^{stab}, is defined as
            // F_i^{stab} = \frac{\partial}{\partial x_j}(u^h_j u^h_i) + Gr_i Pr^2 T^h + \frac{Pr}{Da} u^h_i
            for(Plato::OrdinalType tNode = 0; tNode < mNumNodesPerCell; tNode++)
            {
                for(Plato::OrdinalType tDimI = 0; tDimI < mNumSpatialDims; tDimI++)
                {
                    auto tDofIndex = (mNumSpatialDims * tNode) + tDimI;
                    for(Plato::OrdinalType tDimJ = 0; tDimJ < mNumSpatialDims; tDimJ++)
                    {
                        aResult(aCellOrdinal, tDofIndex) += static_cast<Plato::Scalar>(0.5) * tTimeStepWS(aCellOrdinal, tNode) *
                            tCellVolume(aCellOrdinal) * tBasisFunctions(tNode) * ( tGradient(aCellOrdinal, tNode, tDimJ) *
                                tPrevVelGP(tDimJ) ) * tStabForce(aCellOrdinal, tDimI);
                    }
                }
            }

            // apply time step multiplier to internal force plus stabilized force vector,
            // i.e. F = \Delta{t} * \left( F_i^{int} + F_i^{stab} \right)
            for(Plato::OrdinalType tNode = 0; tNode < mNumNodesPerCell; tNode++)
            {
                for(Plato::OrdinalType tDimI = 0; tDimI < mNumSpatialDims; tDimI++)
                {
                    auto tDofIndex = (mNumSpatialDims * tNode) + tDimI;
                        aResult(aCellOrdinal, tDofIndex) *= tTimeStepWS(aCellOrdinal, tNode);
                }
            }

            // calculate inertial force integral, which are defined as
            // \int_{Omega_e} N_u^a (u^h_\ast) d\Omega_e
            tIntrplVectorField(aCellOrdinal, tBasisFunctions, tPredictorWS, tPredictorGP);
            tCalculateInertialForce(aCellOrdinal, tCellVolume, tBasisFunctions, tPredictorGP, aResult);

        }, "velocity predictor residual");
    }

   void evaluateBoundary
   (const StateWorkSets                & aWorkSets,
    Plato::ScalarMultiVectorT<ResultT> & aResult)
   {
       if( mNeumannLoads != nullptr )
       {
           // set input state worksets
           auto tConfigWS   = aWorkSets.configuration();
           auto tControlWS  = aWorkSets.control();
           auto tPrevVelWS  = aWorkSets.previousVelocity();
           auto tTimeStepWS = aWorkSets.timeStep();

           // set output force
           auto tNumCells = mSpatialDomain.numCells();
           Plato::ScalarMultiVectorT<ResultT> tNeumannForce("neumann force", tNumCells, mNumVelDofsPerCell);
           mNeumannLoads.get( mSpatialDomain, tPrevVelWS, tControlWS, tConfigWS, tNeumannForce, -1.0 );

           Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
           {
               for(Plato::OrdinalType tNode = 0; tNode < mNumNodesPerCell; tNode++)
               {
                   for(Plato::OrdinalType tDof = 0; tDof < mNumVelDofsPerNode; tDof++)
                   {
                       auto tDofIndex = (mNumVelDofsPerNode * tNode) + tDof;
                       aResult(aCellOrdinal, tDofIndex) += tTimeStepWS(aCellOrdinal, tNode) * tNeumannForce(aCellOrdinal, tDofIndex);
                   }
               }
           }, "add external force contribution to residual");
       }
   }

private:
   void setPenaltyModel
   (Teuchos::ParameterList & aInputs)
   {
       if(aProblemParams.isSublist("Hyperbolic"))
       {
           auto tHyperbolicList = aProblemParams.sublist("Hyperbolic");
           if(tHyperbolicList.isSublist("Penalty Function"))
           {
               auto tPenaltyFuncList = tHyperbolicList.sublist("Penalty Function");
               mGrNumExponent = tPenaltyFuncList.get<Plato::Scalar>("Grashof Number Penalty Exponent", 3.0);
               mPrNumConvexityParam = tPenaltyFuncList.get<Plato::Scalar>("Prandtl Number Convexity Parameter", 0.5);
               mBrinkmanConvexityParam = tPenaltyFuncList.get<Plato::Scalar>("Brinkman Convexity Parameter", 0.5);
           }
       }
       else
       {
           THROWERR("'Hyperbolic' sublist is not defined.")
       }
   }

    void setDimensionlessProperties
    (Teuchos::ParameterList & aInputs)
    {
        if(aInputs.isSublist("Dimensionless Properties"))
        {
            auto tSublist = aInputs.sublist("Dimensionless Properties");
            mDaNum = Plato::parse_dimensionless_property<Plato::Scalar>(tSublist, "Darcy Number");
            mPrNum = Plato::parse_dimensionless_property<Plato::Scalar>(tSublist, "Prandtl Number");

            auto tGrNum = Plato::parse_dimensionless_property<Teuchos::Array<Plato::Scalar>>(tSublist, "Grashof Number");
            if(tGrNum.size() != mNumSpatialDims)
            {
                THROWERR("Grashof Number array length should match the number of spatial dimensions.")
            }

            auto tHostGrNum = Kokkos::create_mirror(mGrNum);
            for(decltype(mNumSpatialDims) tDim = 0; tDim < mNumSpatialDims; tDim++)
            {
                tHostGrNum(tDim) = tGrNum[tDim];
            }
            Kokkos::deep_copy(mGrNum, tHostGrNum);
        }
        else
        {
            THROWERR("'Dimensionless Properties' sublist is not defined.")
        }
    }
};
// class VelocityPredictorResidual

template<typename PhysicsT, typename EvaluationT>
class VelocityCorrectorResidual :
    public Plato::Hyperbolic::FluidMechanics::AbstractVectorFunctionBase<PhysicsT, EvaluationT>
{
private:
    static constexpr auto mNumSpatialDims       = PhysicsT::SimplexT::mNumSpatialDims;         /*!< number of spatial dimensions */
    static constexpr auto mNumNodesPerCell      = PhysicsT::SimplexT::mNumNodesPerCell;        /*!< number of nodes per cell */
    static constexpr auto mNumVelDofsPerCell    = PhysicsT::SimplexT::mNumMomentumDofsPerCell; /*!< number of momentum dofs per cell */
    static constexpr auto mNumPressDofsPerCell  = PhysicsT::SimplexT::mNumMassDofsPerCell;     /*!< number of mass dofs per cell */
    static constexpr auto mNumVelDofsPerNode    = PhysicsT::SimplexT::mNumMomentumDofsPerNode; /*!< number of momentum dofs per node */
    static constexpr auto mNumPressDofsPerNode  = PhysicsT::SimplexT::mNumMassDofsPerNode;     /*!< number of mass dofs per node */
    static constexpr auto mNumConfigDofsPerCell = mNumSpatialDims * mNumNodesPerCell;          /*!< number of configuration degrees of freedom per cell */

    using ResultT    = typename EvaluationT::ResultScalarType;
    using ConfigT    = typename EvaluationT::ConfigScalarType;
    using CurPressT  = typename EvaluationT::CurrentMassScalarType;
    using PrevVelT   = typename EvaluationT::PreviousMomentumScalarType;
    using PrevPressT = typename EvaluationT::PreviousMassScalarType;
    using CorrectorT = typename EvaluationT::MomentumCorrectorScalarType;

    Plato::DataMap& mDataMap;                   /*!< output database */
    const Plato::SpatialDomain& mSpatialDomain; /*!< Plato spatial model */

    Plato::LinearTetCubRuleDegreeOne<mNumSpatialDims> mCubatureRule;

    using StateWorkSets = Plato::Hyperbolic::FluidMechanics::WorkSets<PhysicsT, EvaluationT>;

    Plato::Scalar mThetaTwo = 0.0;

public:
    VelocityCorrectorResidual
    (const Plato::SpatialDomain & aDomain,
     Plato::DataMap             & aDataMap,
     Teuchos::ParameterList     & aInputs) :
         mDataMap(aDataMap),
         mSpatialDomain(aDomain),
         mCubatureRule(Plato::LinearTetCubRuleDegreeOne<mNumSpatialDims>())
    {
        if(aInputs.isSublist("Time Integration"))
        {
            auto tTimeIntegration = aInputs.sublist("Time Integration");
            mThetaTwo = tTimeIntegration.get<Plato::Scalar>("Time Step Multiplier Theta 2", 0.0);
        }
    }

    virtual ~VelocityCorrectorResidual(){}

    void evaluate
    (const StateWorkSets                & aWorkSets,
     Plato::ScalarMultiVectorT<ResultT> & aResult) const
    {
        auto tNumCells = mSpatialDomain.numCells();
        Plato::ScalarVectorT<ConfigT>    tCellVolume("cell weight", tNumCells);
        Plato::ScalarVectorT<CurPressT>  tCurPressGP("current pressure at Gauss point", tNumCells);
        Plato::ScalarVectorT<PrevPressT> tPrevPressGP("previous pressure at Gauss point", tNumCells);

        Plato::ScalarMultiVectorT<ResultT>    tStabForce("stabilized force", tNumCells, mNumSpatialDims);
        Plato::ScalarMultiVectorT<PrevVelT>   tPrevVelGP("previous velocity at Gauss point", tNumCells, mNumSpatialDims);
        Plato::ScalarMultiVectorT<CorrectorT> tCorrectorGP("corrector at Gauss point", tNumCells, mNumVelDofsPerNode);

        Plato::ScalarArray3DT<ConfigT> tGradient("cell gradient", tNumCells, mNumNodesPerCell, mNumSpatialDims);

        // set local functors
        Plato::ComputeGradientWorkset<mNumSpatialDims> tComputeGradient;
        Plato::ProjectToNode<mNumSpatialDims, mNumVelDofsPerNode> tCalculateInertialForce;
        Plato::InterpolateFromNodal<mNumSpatialDims, mNumVelDofsPerNode, 0/*offset*/, mNumSpatialDims> tIntrplVectorField;
        Plato::InterpolateFromNodal<mNumSpatialDims, mNumPressDofsPerNode, 0/*offset*/, mNumSpatialDims> tIntrplScalarField;

        // set input state worksets
        auto tConfigWS    = aWorkSets.configuration();
        auto tTimeStepWS  = aWorkSets.timeStep();
        auto tPrevVelWS   = aWorkSets.previousVelocity();
        auto tCurPressWS  = aWorkSets.currentPressure();
        auto tPrevPressWS = aWorkSets.previousPressure();
        auto tCorrectorWS = aWorkSets.corrector();

        // transfer member data to device
        auto tThetaTwo = mThetaTwo;

        auto tCubWeight = mCubatureRule->getCubWeight();
        auto tBasisFunctions = mCubatureRule->getBasisFunctions();
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
        {
            tComputeGradient(aCellOrdinal, tGradient, tConfigWS, tCellVolume);
            tCellVolume(aCellOrdinal) *= tCubWeight;

            // calculate pressure gradient
            tIntrplScalarField(aCellOrdinal, tBasisFunctions, tCurPressWS, tCurPressGP);
            tIntrplScalarField(aCellOrdinal, tBasisFunctions, tPrevPressWS, tPrevPressGP);

            // calculate pressure gradient integral, which is defined as
            //  \int_{\Omega_e} N_u^a \frac{\partial p^{n+\theta_2}}{\partial x_i} d\Omega_e,
            //  where p^{n+\theta_2} = \frac{\partial p^{n-1}}{\partial x_i} + \theta_2
            //  \frac{\partial\delta{p}}{\partial x_i} and \delta{p} = p^n - p^{n-1}
            // NOTE: THE STABILIZING TERM IMPLEMENTED HEREIN USES p^{n+\theta_2} IN THE
            // STABILIZATION TERMS; HOWEVER, THE BOOK MAGICALLY DROPS p^{n+\theta_2} FOR
            // p^{n-1}, STUDY THIS BEHAVIOR AS YOU TEST THE FORMULATION
            for(Plato::OrdinalType tNode = 0; tNode < mNumNodesPerCell; tNode++)
            {
                for(Plato::OrdinalType tDim = 0; tDim < mNumSpatialDims; tDim++)
                {
                    tStabForce(aCellOrdinal, tDim) += ( tGradient(aCellOrdinal, tNode, tDim) *
                        tPrevPressGP(aCellOrdinal) ) + ( tThetaTwo * ( tGradient(aCellOrdinal, tNode, tDim) *
                            ( tCurPressGP(aCellOrdinal) - tPrevPressGP(aCellOrdinal) ) ) );

                    auto tDofIndex = (mNumSpatialDims * tNode) + tDim;
                    aResult(aCellOrdinal, tDofIndex) += tCellVolume(aCellOrdinal) *
                        tBasisFunctions(tNode) * tStabForce(aCellOrdinal, tDim);
                }
            }

            // calculate stabilizing force integral, which is defined as
            // \int_{\Omega_e} \left( \frac{\partial N_u^a}{\partial x_k} u_k \right) * F_i^{stab} d\Omega_e
            // where the stabilizing force, F_i^{stab}, is defined as
            // F_i^{stab} = \frac{\partial p^{n+\theta_2}}{\partial x_i} d\Omega_e
            tIntrplVectorField(aCellOrdinal, tBasisFunctions, tPrevVelWS, tPrevVelGP);
            for(Plato::OrdinalType tNode = 0; tNode < mNumNodesPerCell; tNode++)
            {
                for(Plato::OrdinalType tDimI = 0; tDimI < mNumSpatialDims; tDimI++)
                {
                    auto tDofIndex = (mNumSpatialDims * tNode) + tDimI;
                    for(Plato::OrdinalType tDimJ = 0; tDimJ < mNumSpatialDims; tDimJ++)
                    {
                        aResult(aCellOrdinal, tDofIndex) += static_cast<Plato::Scalar>(0.5) * tTimeStepWS(aCellOrdinal, tNode) *
                            tCellVolume(aCellOrdinal) * tBasisFunctions(tNode) * ( tGradient(aCellOrdinal, tNode, tDimJ) *
                                tPrevVelGP(tDimJ) ) * tStabForce(aCellOrdinal, tDimI);
                    }
                }
            }

            // apply time step multiplier to internal force plus stabilized force vector,
            // i.e. F = \Delta{t} * \left( F_i^{int} + F_i^{stab} \right)
            for(Plato::OrdinalType tNode = 0; tNode < mNumNodesPerCell; tNode++)
            {
                for(Plato::OrdinalType tDimI = 0; tDimI < mNumSpatialDims; tDimI++)
                {
                    auto tDofIndex = (mNumSpatialDims * tNode) + tDimI;
                        aResult(aCellOrdinal, tDofIndex) *= tTimeStepWS(aCellOrdinal, tNode);
                }
            }

            // calculate inertial force integral, which are defined as
            // \int_{Omega_e} N_u^a (u^h_{**}) d\Omega_e
            tIntrplVectorField(aCellOrdinal, tBasisFunctions, tCorrectorWS, tCorrectorGP);
            tCalculateInertialForce(aCellOrdinal, tCellVolume, tBasisFunctions, tCorrectorGP, aResult);
        }, "velocity corrector residual");
    }

    void evaluateBoundary
    (const StateWorkSets                & aWorkSets,
     Plato::ScalarMultiVectorT<ResultT> & aResult)
    { return; }
};
// class VelocityCorrectorResidual

template<typename PhysicsT, typename EvaluationT>
class TemperatureIncrementResidual :
    public Plato::Hyperbolic::FluidMechanics::AbstractVectorFunctionBase<PhysicsT, EvaluationT>
{
private:
    static constexpr auto mNumSpatialDims       = PhysicsT::SimplexT::mNumSpatialDims;         /*!< number of spatial dimensions */
    static constexpr auto mNumNodesPerCell      = PhysicsT::SimplexT::mNumNodesPerCell;        /*!< number of nodes per cell */
    static constexpr auto mNumVelDofsPerCell    = PhysicsT::SimplexT::mNumMomentumDofsPerCell; /*!< number of momentum dofs per cell */
    static constexpr auto mNumTempDofsPerCell   = PhysicsT::SimplexT::mNumEnergyDofsPerCell;   /*!< number of energy dofs per cell */
    static constexpr auto mNumVelDofsPerNode    = PhysicsT::SimplexT::mNumMomentumDofsPerNode; /*!< number of momentum dofs per node */
    static constexpr auto mNumTempDofsPerNode   = PhysicsT::SimplexT::mNumEnergyDofsPerNode;   /*!< number of energy dofs per node */
    static constexpr auto mNumConfigDofsPerCell = mNumSpatialDims * mNumNodesPerCell;          /*!< number of configuration degrees of freedom per cell */

    using ResultT   = typename EvaluationT::ResultScalarType;
    using ConfigT   = typename EvaluationT::ConfigScalarType;
    using ControlT  = typename EvaluationT::ControlScalarType;
    using CurTempT  = typename EvaluationT::CurrentEnergyScalarType;
    using PrevVelT  = typename EvaluationT::PreviousMomentumScalarType;
    using PrevTempT = typename EvaluationT::PreviousEnergyScalarType;

    Plato::DataMap& mDataMap;                   /*!< output database */
    const Plato::SpatialDomain& mSpatialDomain; /*!< Plato spatial model */

    Plato::LinearTetCubRuleDegreeOne<mNumSpatialDims> mCubatureRule;                   /*!< integration rule */
    std::shared_ptr<Plato::NaturalBCs<mNumSpatialDims, mNumVelDofsPerNode>> mHeatFlux; /*!< heat flux evaluator */

    using StateWorkSets = Plato::Hyperbolic::FluidMechanics::WorkSets<PhysicsT, EvaluationT>;

    Plato::Scalar mHeatSource                = 0.0;
    Plato::Scalar mFluidDomainCutOff         = 0.5;
    Plato::Scalar mCharacteristicLength      = 1.0;
    Plato::Scalar mSolidThermalDiffusivity   = 1.0;
    Plato::Scalar mFluidThermalDiffusivity   = 1.0;
    Plato::Scalar mReferenceTempDifference   = 1.0;
    Plato::Scalar mFluidThermalConductivity  = 1.0;
    Plato::Scalar mDiffusivityConvexityParam = 0.5;


public:
    TemperatureIncrementResidual
    (const Plato::SpatialDomain & aDomain,
     Plato::DataMap             & aDataMap,
     Teuchos::ParameterList     & aInputs) :
         mDataMap(aDataMap),
         mSpatialDomain(aDomain),
         mCubatureRule(Plato::LinearTetCubRuleDegreeOne<mNumSpatialDims>()),
         mHeatSource(Plato::ScalarVector("heat source", mNumSpatialDims))
    {
        if(aInputs.isSublist("Heat Flux Boundary Conditions"))
        {
            auto tSublist = aInputs.sublist("Heat Flux Boundary Conditions");
            mHeatFlux = std::make_shared<Plato::NaturalBCs<mNumSpatialDims, mNumVelDofsPerNode>>(tSublist);
        }
    }

    virtual ~TemperatureIncrementResidual(){}

    void evaluate
    (const StateWorkSets                & aWorkSets,
     Plato::ScalarMultiVectorT<ResultT> & aResult) const
    {
        // set local data
        auto tNumCells = mSpatialDomain.numCells();
        Plato::ScalarVectorT<ConfigT>   tCellVolume("cell weight", tNumCells);
        Plato::ScalarVectorT<PrevTempT> tCurTempGP("current temperature at Gauss point", tNumCells);
        Plato::ScalarVectorT<PrevTempT> tPrevTempGP("previous temperature at Gauss point", tNumCells);

        Plato::ScalarMultiVectorT<ResultT>  tStabForce("stabilized force", tNumCells, mNumNodesPerCell);
        Plato::ScalarMultiVectorT<PrevVelT> tPrevVelGP("previous velocity at Gauss point", tNumCells, mNumSpatialDims);
        Plato::ScalarMultiVectorT<ResultT>  tPrevThermalGradGP("previous thermal gradient at Gauss point", tNumCells, mNumSpatialDims);

        Plato::ScalarArray3DT<ConfigT> tGradient("cell gradient", tNumCells, mNumNodesPerCell, mNumSpatialDims);

        // set local functors
        Plato::ComputeGradientWorkset<mNumSpatialDims> tComputeGradient;
        Plato::ProjectToNode<mNumSpatialDims, mNumVelDofsPerNode> tCalculateInertialForce;
        Plato::InterpolateFromNodal<mNumSpatialDims, mNumVelDofsPerNode, 0/*offset*/, mNumSpatialDims> tIntrplVectorField;
        Plato::InterpolateFromNodal<mNumSpatialDims, mNumTempDofsPerNode, 0/*offset*/, mNumSpatialDims> tIntrplScalarField;

        // set input state worksets
        auto tConfigWS   = aWorkSets.configuration();
        auto tControlWS  = aWorkSets.control();
        auto tTimeStepWS = aWorkSets.timeStep();
        auto tPrevVelWS  = aWorkSets.previousVelocity();
        auto tCurTempWS  = aWorkSets.currentTemperature();
        auto tPrevTempWS = aWorkSets.previousTemperature();

        // transfer member data to device
        auto tHeatSource                = mHeatSource;
        auto tFluidDomainCutOff         = mFluidDomainCutOff;
        auto tCharacteristicLength      = mCharacteristicLength;
        auto tFluidThermalDiffusivity   = mFluidThermalDiffusivity;
        auto tSolidThermalDiffusivity   = mSolidThermalDiffusivity;
        auto tReferenceTempDifference   = mReferenceTempDifference;
        auto tFluidThermalConductivity  = mFluidThermalConductivity;
        auto tDiffusivityConvexityParam = mDiffusivityConvexityParam;

        auto tCubWeight = mCubatureRule->getCubWeight();
        auto tBasisFunctions = mCubatureRule->getBasisFunctions();
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
        {
            tComputeGradient(aCellOrdinal, tGradient, tConfigWS, tCellVolume);
            tCellVolume(aCellOrdinal) *= tCubWeight;

            // calculate convective force integral, which is defined as
            // \int_{\Omega_e} N_T^a \left( u^h_i \frac{\partial T^h}{\partial x_i} \right) d\Omega_e
            tIntrplVectorField(aCellOrdinal, tBasisFunctions, tPrevVelWS, tPrevVelGP);
            tIntrplScalarField(aCellOrdinal, tBasisFunctions, tPrevTempWS, tPrevTempGP);
            for(Plato::OrdinalType tNode = 0; tNode < mNumNodesPerCell; tNode++)
            {
                for(Plato::OrdinalType tDim = 0; tDim < mNumSpatialDims; tDim++)
                {
                    tStabForce(aCellOrdinal, tNode) += ( tGradient(aCellOrdinal, tNode, tDim) *
                        tPrevVelGP(aCellOrdinal, tDim) ) * tPrevTempGP(aCellOrdinal);

                    aResult(aCellOrdinal, tNode) += tCellVolume(aCellOrdinal) * tBasisFunctions(tNode) *
                        ( ( tGradient(aCellOrdinal, tNode, tDim) * tPrevVelGP(aCellOrdinal, tDim) ) * tPrevTempGP(aCellOrdinal) );
                }
            }

            // calculate penalized thermal diffusivity properties \pi(\theta), which is defined as
            // \pi(\theta) = \frac{\theta*( \hat{\alpha}(\mathbf{x})(1-q_{\pi}) - 1 ) + 1}{\hat{\alpha}(\mathbf{x})(1+q_{\pi})}
            ControlT tDensity = Plato::cell_density<mNumNodesPerCell>(aCellOrdinal, tControlWS);
            auto tDiffusivityRatio = tDensity >= tFluidDomainCutOff ? 1.0 : tSolidThermalDiffusivity / tFluidThermalDiffusivity;
            ControlT tNumerator = ( tDensity * ( (tDiffusivityRatio * (1.0 - tDiffusivityConvexityParam)) - 1.0 ) ) + 1.0;
            ControlT tDenominator = tDiffusivityRatio * (1.0 + (tDiffusivityConvexityParam * tDensity) );
            ControlT tPenalizedThermalDiff = tNumerator / tDenominator;

            // calculate penalized thermal gradient, which is defined as
            // \pi_T(\theta) \frac{\partial T^h}{\partial x_i} = \pi_T(\theta)\partial_i T^h
            for(Plato::OrdinalType tNode = 0; tNode < mNumNodesPerCell; tNode++)
            {
                for(Plato::OrdinalType tDim = 0; tDim < mNumSpatialDims; tDim++)
                {
                    tPrevThermalGradGP(aCellOrdinal, tDim) += tPenalizedThermalDiff *
                        tGradient(aCellOrdinal, tNode, tDim) * tPrevTempGP(aCellOrdinal);
                }
            }

            // calculate diffusive force integral, which is defined as
            // int_{\Omega_e} \frac{partial N_T^a}{\partial x_i} \left(\pi_T(\theta)\partial_i T^h\right) d\Omega_e
            for(Plato::OrdinalType tNode = 0; tNode < mNumNodesPerCell; tNode++)
            {
                for(Plato::OrdinalType tDim = 0; tDim < mNumSpatialDims; tDim++)
                {
                    aResult(aCellOrdinal, tNode) += tCellVolume(aCellOrdinal) * tBasisFunctions(tNode) *
                        ( tGradient(aCellOrdinal, tNode, tDim) * tPrevThermalGradGP(aCellOrdinal, tDim) );
                }
            }

            // calculate heat source integral, which is defined as
            // \int_{Omega_e} N_T^a (\beta Q_i) d\Omega_e
            auto tHeatSourceConstant = tDensity >= tFluidDomainCutOff ? 0.0 :
                tCharacteristicLength * tCharacteristicLength / (tFluidThermalConductivity * tReferenceTempDifference);
            for(Plato::OrdinalType tNode = 0; tNode < mNumNodesPerCell; tNode++)
            {
                tStabForce(aCellOrdinal, tNode) -= tHeatSourceConstant * tHeatSource;
                for(Plato::OrdinalType tDim = 0; tDim < mNumSpatialDims; tDim++)
                {
                    aResult(aCellOrdinal, tNode) -= tCellVolume(aCellOrdinal) *
                        tBasisFunctions(tNode) * tHeatSourceConstant * tHeatSource;
                }
            }

            // calculate stabilizing force integral, which is defined as
            // \int_{\Omega_e} \frac{\partial N_T^a}{\partial x_k}u_k F^{stab} d\Omega_e
            // where F^{stab} = u_i\frac{\partial T^h}{\partial x_i} - \beta Q
            for(Plato::OrdinalType tNode = 0; tNode < mNumNodesPerCell; tNode++)
            {
                for(Plato::OrdinalType tDim = 0; tDim < mNumSpatialDims; tDim++)
                {
                    aResult(aCellOrdinal, tNode) += static_cast<Plato::Scalar>(0.5) * tTimeStepWS(aCellOrdinal, tNode) *
                        tCellVolume(aCellOrdinal) * tBasisFunctions(tNode) * ( tGradient(aCellOrdinal, tNode, tDim) *
                            tPrevVelGP(aCellOrdinal, tDim) ) * tStabForce(aCellOrdinal, tNode);
                }
            }

            // apply time step multiplier to internal force plus stabilizing force vectors,
            // i.e. F = \Delta{t} * \left( F_i^{int} + F_i^{stab} \right)
            for(Plato::OrdinalType tNode = 0; tNode < mNumNodesPerCell; tNode++)
            {
                aResult(aCellOrdinal, tNode) *= tTimeStepWS(aCellOrdinal, tNode);
            }

            // calculate inertial force integral, which are defined as
            // \int_{Omega_e} N_T^a (T^h) d\Omega_e
            tIntrplVectorField(aCellOrdinal, tBasisFunctions, tCurTempWS, tCurTempGP);
            tCalculateInertialForce(aCellOrdinal, tCellVolume, tBasisFunctions, tCurTempGP, aResult);

        }, "temperature increment residual");
    }

    void evaluateBoundary
    (const StateWorkSets                & aWorkSets,
     Plato::ScalarMultiVectorT<ResultT> & aResult)
    {
        if( mNeumannLoads != nullptr )
        {
            // set input state worksets
            auto tConfigWS   = aWorkSets.configuration();
            auto tControlWS  = aWorkSets.control();
            auto tTimeStepWS = aWorkSets.timeStep();
            auto tPrevTempWS = aWorkSets.previousTemperature();

            // set output force
            auto tNumCells = mSpatialDomain.numCells();
            Plato::ScalarMultiVectorT<ResultT> tHeatFlux("heat flux", tNumCells, mNumNodesPerCell);
            mNeumannLoads.get( mSpatialDomain, tPrevTempWS, tControlWS, tConfigWS, tHeatFlux, -1.0 );

            Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
            {
                for(Plato::OrdinalType tDof = 0; tDof < mNumTempDofsPerCell; tDof++)
                {
                    aResult(aCellOrdinal, tDof) += tTimeStepWS(aCellOrdinal, tDof) * tHeatFlux(aCellOrdinal, tDof);
                }
            }, "add heat flux to residual");
        }
    }
};
// class TemperatureIncrementResidual

/******************************************************************************/
/*! vector function class

   This class takes as a template argument a vector function in the form:

   \f$ F = F(\phi, U^k, P^k, T^k, X) \f$

   and manages the evaluation of the function and derivatives with respect to
   control, \f$\phi\f$, momentum state, \f$ U^k \f$, mass state, \f$ P^k \f$,
   energy state, \f$ T^k \f$, and configuration, \f$ X \f$.

*/
/******************************************************************************/
template<typename PhysicsT>
class VectorFunction
{
private:
    static constexpr auto mNumControlPerNode    = PhysicsT::SimplexT::mNumControl;             /*!< number of design variable fields */
    static constexpr auto mNumSpatialDims       = PhysicsT::SimplexT::mNumSpatialDims;         /*!< number of spatial dimensions */
    static constexpr auto mNumNodesPerCell      = PhysicsT::SimplexT::mNumNodesPerCell;        /*!< number of nodes per cell */
    static constexpr auto mNumPressDofsPerCell  = PhysicsT::SimplexT::mNumMassDofsPerCell;     /*!< number of mass dofs per cell */
    static constexpr auto mNumTempDofsPerCell   = PhysicsT::SimplexT::mNumEnergyDofsPerCell;   /*!< number of energy dofs per cell */
    static constexpr auto mNumVelDofsPerCell    = PhysicsT::SimplexT::mNumMomentumDofsPerCell; /*!< number of momentum dofs per cell */
    static constexpr auto mNumPressDofsPerNode  = PhysicsT::SimplexT::mNumMassDofsPerNode;     /*!< number of mass dofs per node */
    static constexpr auto mNumTempDofsPerNode   = PhysicsT::SimplexT::mNumEnergyDofsPerNode;   /*!< number of energy dofs per node */
    static constexpr auto mNumVelDofsPerNode    = PhysicsT::SimplexT::mNumMomentumDofsPerNode; /*!< number of momentum dofs per node */
    static constexpr auto mNumConfigDofsPerCell = mNumSpatialDims * mNumNodesPerCell;          /*!< number of configuration degrees of freedom per cell */

    static constexpr auto mNumTimeStepsPerNode  = 1; /*!< number of time step dofs per node */

    // forward automatic differentiation types
    using Residual      = typename Plato::Hyperbolic::FluidMechanics::Evaluation<typename PhysicsT::SimplexT>::Residual;
    using GradConfig    = typename Plato::Hyperbolic::FluidMechanics::Evaluation<typename PhysicsT::SimplexT>::GradConfig;
    using GradControl   = typename Plato::Hyperbolic::FluidMechanics::Evaluation<typename PhysicsT::SimplexT>::GradControl;
    using GradCurrVel   = typename Plato::Hyperbolic::FluidMechanics::Evaluation<typename PhysicsT::SimplexT>::GradMomentumCurr;
    using GradPrevVel   = typename Plato::Hyperbolic::FluidMechanics::Evaluation<typename PhysicsT::SimplexT>::GradMomentumPrev;
    using GradCurrTemp  = typename Plato::Hyperbolic::FluidMechanics::Evaluation<typename PhysicsT::SimplexT>::GradEnergyCurr;
    using GradPrevTemp  = typename Plato::Hyperbolic::FluidMechanics::Evaluation<typename PhysicsT::SimplexT>::GradEnergyPrev;
    using GradCurrPress = typename Plato::Hyperbolic::FluidMechanics::Evaluation<typename PhysicsT::SimplexT>::GradMassCurr;
    using GradPrevPress = typename Plato::Hyperbolic::FluidMechanics::Evaluation<typename PhysicsT::SimplexT>::GradMassPrev;
    using GradPredictor = typename Plato::Hyperbolic::FluidMechanics::Evaluation<typename PhysicsT::SimplexT>::GradPredictor;
    using GradCorrector = typename Plato::Hyperbolic::FluidMechanics::Evaluation<typename PhysicsT::SimplexT>::GradCorrector;

    // element residual functions
    using ResidualFunc      = std::shared_ptr<Plato::Hyperbolic::FluidMechanics::AbstractVectorFunctionBase<typename PhysicsT::SimplexT, Residual>>;
    using GradConfigFunc    = std::shared_ptr<Plato::Hyperbolic::FluidMechanics::AbstractVectorFunctionBase<typename PhysicsT::SimplexT, GradConfig>>;
    using GradControlFunc   = std::shared_ptr<Plato::Hyperbolic::FluidMechanics::AbstractVectorFunctionBase<typename PhysicsT::SimplexT, GradControl>>;
    using GradCurrVelFunc   = std::shared_ptr<Plato::Hyperbolic::FluidMechanics::AbstractVectorFunctionBase<typename PhysicsT::SimplexT, GradCurrVel>>;
    using GradPrevVelFunc   = std::shared_ptr<Plato::Hyperbolic::FluidMechanics::AbstractVectorFunctionBase<typename PhysicsT::SimplexT, GradPrevVel>>;
    using GradCurrTempFunc  = std::shared_ptr<Plato::Hyperbolic::FluidMechanics::AbstractVectorFunctionBase<typename PhysicsT::SimplexT, GradCurrTemp>>;
    using GradPrevTempFunc  = std::shared_ptr<Plato::Hyperbolic::FluidMechanics::AbstractVectorFunctionBase<typename PhysicsT::SimplexT, GradPrevTemp>>;
    using GradCurrPressFunc = std::shared_ptr<Plato::Hyperbolic::FluidMechanics::AbstractVectorFunctionBase<typename PhysicsT::SimplexT, GradCurrPress>>;
    using GradPrevPressFunc = std::shared_ptr<Plato::Hyperbolic::FluidMechanics::AbstractVectorFunctionBase<typename PhysicsT::SimplexT, GradPrevPress>>;
    using GradPredictorFunc = std::shared_ptr<Plato::Hyperbolic::FluidMechanics::AbstractVectorFunctionBase<typename PhysicsT::SimplexT, GradPredictor>>;
    using GradCorrectorFunc = std::shared_ptr<Plato::Hyperbolic::FluidMechanics::AbstractVectorFunctionBase<typename PhysicsT::SimplexT, GradCorrector>>;

    // element vector functions per element block, i.e. domain
    std::unordered_map<std::string, ResidualFunc>      mResidualFuncs;
    std::unordered_map<std::string, GradConfigFunc>    mGradConfigFuncs;
    std::unordered_map<std::string, GradControlFunc>   mGradControlFuncs;
    std::unordered_map<std::string, GradCurrVelFunc>   mGradCurrVelFuncs;
    std::unordered_map<std::string, GradPrevVelFunc>   mGradPrevVelFuncs;
    std::unordered_map<std::string, GradCurrTempFunc>  mGradCurrTempFuncs;
    std::unordered_map<std::string, GradPrevTempFunc>  mGradPrevTempFuncs;
    std::unordered_map<std::string, GradCurrPressFunc> mGradCurrPressFuncs;
    std::unordered_map<std::string, GradPrevPressFunc> mGradPrevPressFuncs;
    std::unordered_map<std::string, GradPredictorFunc> mGradPredictorFuncs;
    std::unordered_map<std::string, GradPredictorFunc> mGradCorrectorFuncs;

    Plato::DataMap& mDataMap;
    const Plato::SpatialModel& mSpatialModel;

    Plato::NodeCoordinate<mNumSpatialDims> mNodeCoordinate; /*!< node coordinates metadata */

    Plato::VectorEntryOrdinal<mNumSpatialDims, mNumSpatialDims>      mConfigEntryOrdinal;      /*!< configuration local-to-global ID map */
    Plato::VectorEntryOrdinal<mNumSpatialDims, mNumControlPerNode>   mControlEntryOrdinal;     /*!< control local-to-global ID map */
    Plato::VectorEntryOrdinal<mNumSpatialDims, mNumVelDofsPerCell>   mVectorStateEntryOrdinal; /*!< vector state (e.g. velocity) local-to-global ID map */
    Plato::VectorEntryOrdinal<mNumSpatialDims, mNumPressDofsPerCell> mScalarStateEntryOrdinal; /*!< scalar state (e.g. pressure) local-to-global ID map */

    using PrimalStates = Plato::Hyperbolic::FluidMechanics::States;

public:
    /**************************************************************************//**
    * \brief Constructor
    * \param [in] aSpatialModel struct that contains the mesh, meshsets, domains, etc.
    * \param [in] aDataMap      problem-specific data map
    * \param [in] aInputs       Teuchos parameter list with input data
    * \param [in] aProblemType  problem type
    ******************************************************************************/
    VectorFunction
    (const Plato::SpatialModel & aSpatialModel,
     Plato::DataMap & aDataMap,
     Teuchos::ParameterList & aInputs,
     std::string & aName) :
        Plato::WorksetBase<PhysicsT>(aSpatialModel.Mesh),
        mSpatialModel(aSpatialModel),
        mDataMap(aDataMap),
        mConfigEntryOrdinal(Plato::VectorEntryOrdinal<mNumSpatialDims, mNumSpatialDims>(&aSpatialModel.Mesh)),
        mControlEntryOrdinal(Plato::VectorEntryOrdinal<mNumSpatialDims, mNumControlPerNode>(&aSpatialModel.Mesh)),
        mVectorStateEntryOrdinal(Plato::VectorEntryOrdinal<mNumSpatialDims, mNumVelDofsPerCell>(&aSpatialModel.Mesh)),
        mScalarStateEntryOrdinal(Plato::VectorEntryOrdinal<mNumSpatialDims, mNumPressDofsPerCell>(&aSpatialModel.Mesh))
    {
        this->initialize(aDataMap, aInputs, aName);
    }

    /**************************************************************************//**
    * \brief Return total number of momentum degrees of freedom
    ******************************************************************************/
    Plato::OrdinalType size() const
    {
        auto tNumNodes = mSpatialModel.Mesh.nverts();
        return (tNumNodes * mNumVelDofsPerCell);
    }

    Plato::ScalarVector value
    (const Plato::ScalarVector & aControls,
     const PrimalStates & aStates) const
    {
        auto tLength = this->size();
        Plato::ScalarVector tReturnValue("Assembled Residual", tLength);

        // internal force contribution
        auto tFunctionType = Plato::is_valid_function(aStates.function());
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            Plato::Hyperbolic::FluidMechanics::WorkSets<PhysicsT, Residual> tWorkSets(tNumCells);
            this->setWorkSets(tDomain, aControls, aStates, tWorkSets);

            auto tName = tDomain.getDomainName();
            Plato::ScalarMultiVectorT<Residual> tResultWS("Cells Results", tNumCells, PhysicsT::mNumDofsPerCell);
            mResidualFuncs.at(tName)->evaluate(tWorkSets, tResultWS);

            if(tFunctionType == "vector function")
            {
                Plato::assemble_residual<mNumNodesPerCell, mNumVelDofsPerNode>
                    (tDomain, mVectorStateEntryOrdinal, tResultWS, tReturnValue);
            }
            else
            {
                Plato::assemble_residual<mNumNodesPerCell, mNumVelDofsPerNode>
                    (tDomain, mScalarStateEntryOrdinal, tResultWS, tReturnValue);
            }
        }

        // external force contribution
        {
            auto tNumCells = mSpatialModel.Mesh.nverts();
            Plato::Hyperbolic::FluidMechanics::WorkSets<PhysicsT, Residual> tWorkSets(tNumCells);
            this->setWorkSets(aControls, aStates, tWorkSets);

            Plato::ScalarMultiVectorT<Residual> tResultWS("Cells Results", tNumCells, PhysicsT::mNumDofsPerCell);
            mResidualFuncs.begin()->evaluateBoundary(tWorkSets, tResultWS);

            if(tFunctionType == "vector function")
            {
                Plato::assemble_residual<mNumNodesPerCell, mNumVelDofsPerNode>
                    (tNumCells, mVectorStateEntryOrdinal, tResultWS, tReturnValue);
            }
            else
            {
                Plato::assemble_residual<mNumNodesPerCell, mNumVelDofsPerNode>
                    (tNumCells, mScalarStateEntryOrdinal, tResultWS, tReturnValue);
            }
        }

        return tReturnValue;
    }

    Teuchos::RCP<Plato::CrsMatrixType>
    gradientConfig
    (const Plato::ScalarVector & aControls,
     const PrimalStates & aStates) const
    {
        // create return matrix
        auto tMesh = mSpatialModel.Mesh;
        Teuchos::RCP<Plato::CrsMatrixType> tJacobian =
            Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumSpatialDims, mNumVelDofsPerNode>(&tMesh);

        // internal force contribution
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            Plato::Hyperbolic::FluidMechanics::WorkSets<PhysicsT, GradConfig> tWorkSets(tNumCells);
            this->setWorkSets(tDomain, aControls, aStates, tWorkSets);

            auto tName = tDomain.getDomainName();
            Plato::ScalarMultiVectorT<GradConfig> tResultWS("Cells Results", tNumCells, PhysicsT::mNumDofsPerCell);
            mGradConfigFuncs.at(tName)->evaluate(tWorkSets, tResultWS);

            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumSpatialDims, mNumVelDofsPerNode> tJacobianEntryOrdinal(tJacobian, &tMesh);
            Plato::assemble_transpose_jacobian(tDomain, mNumVelDofsPerCell, mNumConfigDofsPerCell, tJacobianEntryOrdinal, tResultWS, tJacobian->entries());
        }

        // external force contribution
        {
            auto tNumCells = mSpatialModel.Mesh.nverts();
            Plato::Hyperbolic::FluidMechanics::WorkSets<PhysicsT, GradConfig> tWorkSets(tNumCells);
            this->setWorkSets(aControls, aStates, tWorkSets);

            Plato::ScalarMultiVectorT<GradConfig> tResultWS("Cells Results", tNumCells, PhysicsT::mNumDofsPerCell);
            mGradConfigFuncs.begin()->evaluateBoundary(tWorkSets, tResultWS);

            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumSpatialDims, mNumVelDofsPerNode> tJacobianEntryOrdinal(tJacobian, &tMesh);
            Plato::assemble_transpose_jacobian(tNumCells, mNumVelDofsPerCell, mNumConfigDofsPerCell, tJacobianEntryOrdinal, tResultWS, tJacobian->entries());
        }

        return tJacobian;
    }

    Teuchos::RCP<Plato::CrsMatrixType>
    gradientControl
    (const Plato::ScalarVector & aControls,
     const PrimalStates & aStates) const
    {
        // create return matrix
        auto tMesh = mSpatialModel.Mesh;
        Teuchos::RCP<Plato::CrsMatrixType> tJacobian =
            Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumControlPerNode, mNumVelDofsPerNode>( &tMesh );

        // internal force contribution
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            Plato::Hyperbolic::FluidMechanics::WorkSets<PhysicsT, GradControl> tWorkSets(tNumCells);
            this->setWorkSets(tDomain, aControls, aStates, tWorkSets);

            auto tName = tDomain.getDomainName();
            Plato::ScalarMultiVectorT<GradControl> tResultWS("Cells Results", tNumCells, PhysicsT::mNumDofsPerCell);
            mGradControlFuncs.at(tName)->evaluate(tWorkSets, tResultWS);

            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumControlPerNode, mNumVelDofsPerNode> tJacobianEntryOrdinal(tJacobian, &tMesh);
            Plato::assemble_transpose_jacobian(tDomain, mNumVelDofsPerCell, mNumNodesPerCell, tJacobianEntryOrdinal, tResultWS, tJacobian->entries());
        }

        // external force contribution
        {
            auto tNumCells = mSpatialModel.Mesh.nverts();
            Plato::Hyperbolic::FluidMechanics::WorkSets<PhysicsT, GradControl> tWorkSets(tNumCells);
            this->setWorkSets(aControls, aStates, tWorkSets);

            Plato::ScalarMultiVectorT<GradControl> tResultWS("Cells Results", tNumCells, PhysicsT::mNumDofsPerCell);
            mGradControlFuncs.begin()->evaluateBoundary(tWorkSets, tResultWS);

            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumControlPerNode, mNumVelDofsPerNode> tJacobianEntryOrdinal(tJacobian, &tMesh);
            Plato::assemble_transpose_jacobian(tNumCells, mNumVelDofsPerCell, mNumNodesPerCell, tJacobianEntryOrdinal, tResultWS, tJacobian->entries());
        }

        return tJacobian;
    }

    Teuchos::RCP<Plato::CrsMatrixType>
    gradientPredictor
    (const Plato::ScalarVector & aControls,
     const PrimalStates & aStates) const
    {
        // create return matrix
        auto tMesh = mSpatialModel.Mesh;
        Teuchos::RCP<Plato::CrsMatrixType> tJacobian =
            Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumVelDofsPerNode, mNumVelDofsPerNode>( &tMesh );

        // internal force contribution
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            Plato::Hyperbolic::FluidMechanics::WorkSets<PhysicsT, GradPredictor> tWorkSets(tNumCells);
            this->setWorkSets(tDomain, aControls, aStates, tWorkSets);

            auto tName = tDomain.getDomainName();
            Plato::ScalarMultiVectorT<GradPredictor> tResultWS("Cells Results", tNumCells, PhysicsT::mNumDofsPerCell);
            mGradPredictorFuncs.at(tName)->evaluate(tWorkSets, tResultWS);

            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumVelDofsPerNode, mNumVelDofsPerNode> tJacobianEntryOrdinal(tJacobian, &tMesh);
            Plato::assemble_transpose_jacobian(tDomain, mNumVelDofsPerCell, mNumVelDofsPerCell, tJacobianEntryOrdinal, tResultWS, tJacobian->entries());
        }

        // external force contribution
        {
            auto tNumCells = mSpatialModel.Mesh.nverts();
            Plato::Hyperbolic::FluidMechanics::WorkSets<PhysicsT, GradPredictor> tWorkSets(tNumCells);
            this->setWorkSets(aControls, aStates, tWorkSets);

            Plato::ScalarMultiVectorT<GradPredictor> tResultWS("Cells Results", tNumCells, PhysicsT::mNumDofsPerCell);
            mGradPredictorFuncs.begin()->evaluateBoundary(tWorkSets);

            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumVelDofsPerNode, mNumVelDofsPerNode> tJacobianEntryOrdinal(tJacobian, &tMesh);
            Plato::assemble_transpose_jacobian(tNumCells, mNumVelDofsPerCell, mNumVelDofsPerCell, tJacobianEntryOrdinal, tResultWS, tJacobian->entries());
        }

        return tJacobian;
    }

    Teuchos::RCP<Plato::CrsMatrixType>
    gradientCorrector
    (const Plato::ScalarVector & aControls,
     const PrimalStates & aStates) const
    {
        // create return matrix
        auto tMesh = mSpatialModel.Mesh;
        Teuchos::RCP<Plato::CrsMatrixType> tJacobian =
            Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumVelDofsPerNode, mNumVelDofsPerNode>( &tMesh );

        // internal force contribution
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            Plato::Hyperbolic::FluidMechanics::WorkSets<PhysicsT, GradCorrector> tWorkSets(tNumCells);
            this->setWorkSets(tDomain, aControls, aStates, tWorkSets);

            auto tName = tDomain.getDomainName();
            Plato::ScalarMultiVectorT<GradCorrector> tResultWS("Cells Results", tNumCells, PhysicsT::mNumDofsPerCell);
            mGradCorrectorFuncs.at(tName)->evaluate(tWorkSets, tResultWS);

            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumVelDofsPerNode, mNumVelDofsPerNode> tJacobianEntryOrdinal(tJacobian, &tMesh);
            Plato::assemble_transpose_jacobian(tDomain, mNumVelDofsPerCell, mNumVelDofsPerCell, tJacobianEntryOrdinal, tResultWS, tJacobian->entries());
        }

        // external force contribution
        {
            auto tNumCells = mSpatialModel.Mesh.nverts();
            Plato::Hyperbolic::FluidMechanics::WorkSets<PhysicsT, GradCorrector> tWorkSets(tNumCells);
            this->setWorkSets(aControls, aStates, tWorkSets);

            Plato::ScalarMultiVectorT<GradCorrector> tResultWS("Cells Results", tNumCells, PhysicsT::mNumDofsPerCell);
            mGradCorrectorFuncs.begin()->evaluateBoundary(tWorkSets, tResultWS);

            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumVelDofsPerNode, mNumVelDofsPerNode> tJacobianEntryOrdinal(tJacobian, &tMesh);
            Plato::assemble_transpose_jacobian(tNumCells, mNumVelDofsPerCell, mNumVelDofsPerCell, tJacobianEntryOrdinal, tResultWS, tJacobian->entries());
        }

        return tJacobian;
    }

    Teuchos::RCP<Plato::CrsMatrixType>
    gradientPreviousVel
    (const Plato::ScalarVector & aControls,
     const PrimalStates & aStates) const
    {
        // create return matrix
        auto tMesh = mSpatialModel.Mesh;
        Teuchos::RCP<Plato::CrsMatrixType> tJacobian =
            Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumVelDofsPerNode, mNumVelDofsPerNode>( &tMesh );

        // internal force contribution
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            Plato::Hyperbolic::FluidMechanics::WorkSets<PhysicsT, GradPrevVel> tWorkSets(tNumCells);
            this->setWorkSets(tDomain, aControls, aStates, tWorkSets);

            auto tName = tDomain.getDomainName();
            Plato::ScalarMultiVectorT<GradPrevVel> tResultWS("Cells Results", tNumCells, PhysicsT::mNumDofsPerCell);
            mGradPrevVelFuncs.at(tName)->evaluate(tWorkSets, tResultWS);

            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumVelDofsPerNode, mNumVelDofsPerNode> tJacobianEntryOrdinal(tJacobian, &tMesh);
            Plato::assemble_transpose_jacobian(tDomain, mNumVelDofsPerCell, mNumVelDofsPerCell, tJacobianEntryOrdinal, tResultWS, tJacobian->entries());
        }

        // external force contribution
        {
            auto tNumCells = mSpatialModel.Mesh.nverts();
            Plato::Hyperbolic::FluidMechanics::WorkSets<PhysicsT, GradPrevVel> tWorkSets(tNumCells);
            this->setWorkSets(aControls, aStates, tWorkSets);

            Plato::ScalarMultiVectorT<GradPrevVel> tResultWS("Cells Results", tNumCells, PhysicsT::mNumDofsPerCell);
            mGradPrevVelFuncs.begin()->evaluateBoundary(tWorkSets, tResultWS);

            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumVelDofsPerNode, mNumVelDofsPerNode> tJacobianEntryOrdinal(tJacobian, &tMesh);
            Plato::assemble_transpose_jacobian(tNumCells, mNumVelDofsPerCell, mNumVelDofsPerCell, tJacobianEntryOrdinal, tResultWS, tJacobian->entries());
        }

        return tJacobian;
    }

    Teuchos::RCP<Plato::CrsMatrixType>
    gradientPreviousPress
    (const Plato::ScalarVector & aControls,
     const PrimalStates & aStates) const
    {
        // create return matrix
        auto tMesh = mSpatialModel.Mesh;
        Teuchos::RCP<Plato::CrsMatrixType> tJacobian =
            Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumPressDofsPerNode, mNumVelDofsPerNode>( &tMesh );

        // internal force contribution
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            Plato::Hyperbolic::FluidMechanics::WorkSets<PhysicsT, GradPrevPress> tWorkSets(tNumCells);
            this->setWorkSets(tDomain, aControls, aStates, tWorkSets);

            auto tName = tDomain.getDomainName();
            Plato::ScalarMultiVectorT<GradPrevPress> tResultWS("Cells Results", tNumCells, PhysicsT::mNumDofsPerCell);
            mGradPrevPressFuncs.at(tName)->evaluate(tWorkSets, tResultWS);

            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumPressDofsPerNode, mNumVelDofsPerNode> tJacobianEntryOrdinal(tJacobian, &tMesh);
            Plato::assemble_transpose_jacobian(tDomain, mNumVelDofsPerCell, mNumPressDofsPerCell, tJacobianEntryOrdinal, tResultWS, tJacobian->entries());
        }

        // external force contribution
        {
            auto tNumCells = mSpatialModel.Mesh.nverts();
            Plato::Hyperbolic::FluidMechanics::WorkSets<PhysicsT, GradPrevPress> tWorkSets(tNumCells);
            this->setWorkSets(aControls, aStates, tWorkSets);

            Plato::ScalarMultiVectorT<GradPrevPress> tResultWS("Cells Results", tNumCells, PhysicsT::mNumDofsPerCell);
            mGradPrevPressFuncs.begin()->evaluateBoundary(tWorkSets, tResultWS);

            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumPressDofsPerNode, mNumVelDofsPerNode> tJacobianEntryOrdinal(tJacobian, &tMesh);
            Plato::assemble_transpose_jacobian(tNumCells, mNumVelDofsPerCell, mNumPressDofsPerCell, tJacobianEntryOrdinal, tResultWS, tJacobian->entries());
        }

        return tJacobian;
    }

    Teuchos::RCP<Plato::CrsMatrixType>
    gradientPreviousTemp
    (const Plato::ScalarVector & aControls,
     const PrimalStates & aStates) const
    {
        // create return matrix
        auto tMesh = mSpatialModel.Mesh;
        Teuchos::RCP<Plato::CrsMatrixType> tJacobian =
            Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumTempDofsPerNode, mNumVelDofsPerNode>( &tMesh );

        // internal force contribution
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            Plato::Hyperbolic::FluidMechanics::WorkSets<PhysicsT, GradPrevTemp> tWorkSets(tNumCells);
            this->setWorkSets(tDomain, aControls, aStates, tWorkSets);

            auto tName = tDomain.getDomainName();
            Plato::ScalarMultiVectorT<GradPrevTemp> tResultWS("Cells Results", tNumCells, PhysicsT::mNumDofsPerCell);
            mGradPrevTempFuncs.at(tName)->evaluate(tWorkSets, tResultWS);

            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumTempDofsPerNode, mNumVelDofsPerNode> tJacobianEntryOrdinal(tJacobian, &tMesh);
            Plato::assemble_transpose_jacobian(tDomain, mNumVelDofsPerCell, mNumTempDofsPerCell, tJacobianEntryOrdinal, tResultWS, tJacobian->entries());
        }

        // external force contribution
        {
            auto tNumCells = mSpatialModel.Mesh.nverts();
            Plato::Hyperbolic::FluidMechanics::WorkSets<PhysicsT, GradPrevTemp> tWorkSets(tNumCells);
            this->setWorkSets(aControls, aStates, tWorkSets);

            Plato::ScalarMultiVectorT<GradPrevTemp> tResultWS("Cells Results", tNumCells, PhysicsT::mNumDofsPerCell);
            mGradPrevTempFuncs.begin()->evaluateBoundary(tWorkSets, tResultWS);

            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumTempDofsPerNode, mNumVelDofsPerNode> tJacobianEntryOrdinal(tJacobian, &tMesh);
            Plato::assemble_transpose_jacobian(tNumCells, mNumVelDofsPerCell, mNumTempDofsPerCell, tJacobianEntryOrdinal, tResultWS, tJacobian->entries());
        }

        return tJacobian;
    }

    Teuchos::RCP<Plato::CrsMatrixType>
    gradientCurrentVel
    (const Plato::ScalarVector & aControls,
     const PrimalStates & aStates) const
    {
        // create return matrix
        auto tMesh = mSpatialModel.Mesh;
        Teuchos::RCP<Plato::CrsMatrixType> tJacobian =
            Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumVelDofsPerNode, mNumVelDofsPerNode>( &tMesh );

        // internal force contribution
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            Plato::Hyperbolic::FluidMechanics::WorkSets<PhysicsT, GradCurrVel> tWorkSets(tNumCells);
            this->setWorkSets(tDomain, aControls, aStates, tWorkSets);

            auto tName = tDomain.getDomainName();
            Plato::ScalarMultiVectorT<GradCurrVel> tResultWS("Cells Results", tNumCells, PhysicsT::mNumDofsPerCell);
            mGradCurrVelFuncs.at(tName)->evaluate(tWorkSets, tResultWS);

            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumVelDofsPerNode, mNumVelDofsPerNode> tJacobianEntryOrdinal(tJacobian, &tMesh);
            Plato::assemble_transpose_jacobian(tDomain, mNumVelDofsPerCell, mNumVelDofsPerCell, tJacobianEntryOrdinal, tResultWS, tJacobian->entries());
        }

        // external force contribution
        {
            auto tNumCells = mSpatialModel.Mesh.nverts();
            Plato::Hyperbolic::FluidMechanics::WorkSets<PhysicsT, GradCurrVel> tWorkSets(tNumCells);
            this->setWorkSets(aControls, aStates, tWorkSets);

            Plato::ScalarMultiVectorT<GradCurrVel> tResultWS("Cells Results", tNumCells, PhysicsT::mNumDofsPerCell);
            mGradCurrVelFuncs.begin()->evaluateBoundary(tWorkSets, tResultWS);

            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumVelDofsPerNode, mNumVelDofsPerNode> tJacobianEntryOrdinal(tJacobian, &tMesh);
            Plato::assemble_transpose_jacobian(tNumCells, mNumVelDofsPerCell, mNumVelDofsPerCell, tJacobianEntryOrdinal, tResultWS, tJacobian->entries());
        }

        return tJacobian;
    }

    Teuchos::RCP<Plato::CrsMatrixType>
    gradientCurrentPress
    (const Plato::ScalarVector & aControls,
     const PrimalStates & aStates) const
    {
        // create return matrix
        auto tMesh = mSpatialModel.Mesh;
        Teuchos::RCP<Plato::CrsMatrixType> tJacobian =
            Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumPressDofsPerNode, mNumVelDofsPerNode>( &tMesh );

        // internal force contribution
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            Plato::Hyperbolic::FluidMechanics::WorkSets<PhysicsT, GradCurrPress> tWorkSets(tNumCells);
            this->setWorkSets(tDomain, aControls, aStates, tWorkSets);

            auto tName = tDomain.getDomainName();
            Plato::ScalarMultiVectorT<GradCurrPress> tResultWS("Cells Results", tNumCells, PhysicsT::mNumDofsPerCell);
            mGradCurrPressFuncs.at(tName)->evaluate(tWorkSets, tResultWS);

            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumPressDofsPerNode, mNumVelDofsPerNode> tJacobianEntryOrdinal(tJacobian, &tMesh);
            Plato::assemble_transpose_jacobian(tDomain, mNumVelDofsPerCell, mNumPressDofsPerCell, tJacobianEntryOrdinal, tResultWS, tJacobian->entries());
        }

        // external force contribution
        {
            auto tNumCells = mSpatialModel.Mesh.nverts();
            Plato::Hyperbolic::FluidMechanics::WorkSets<PhysicsT, GradCurrPress> tWorkSets(tNumCells);
            this->setWorkSets(aControls, aStates, tWorkSets);

            Plato::ScalarMultiVectorT<GradCurrPress> tResultWS("Cells Results", tNumCells, PhysicsT::mNumDofsPerCell);
            mGradCurrPressFuncs.begin()->evaluateBoundary(tWorkSets, tResultWS);

            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumPressDofsPerNode, mNumVelDofsPerNode> tJacobianEntryOrdinal(tJacobian, &tMesh);
            Plato::assemble_transpose_jacobian(tNumCells, mNumVelDofsPerCell, mNumPressDofsPerCell, tJacobianEntryOrdinal, tResultWS, tJacobian->entries());
        }

        return tJacobian;
    }

    Teuchos::RCP<Plato::CrsMatrixType>
    gradientCurrentTemp
    (const Plato::ScalarVector & aControls,
     const PrimalStates & aStates) const
    {
        // create return matrix
        auto tMesh = mSpatialModel.Mesh;
        Teuchos::RCP<Plato::CrsMatrixType> tJacobian =
            Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumTempDofsPerNode, mNumVelDofsPerNode>( &tMesh );

        // internal force contribution
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            Plato::Hyperbolic::FluidMechanics::WorkSets<PhysicsT, GradCurrTemp> tWorkSets(tNumCells);
            this->setWorkSets(tDomain, aControls, aStates, tWorkSets);

            auto tName = tDomain.getDomainName();
            Plato::ScalarMultiVectorT<GradCurrTemp> tResultWS("Cells Results", tNumCells, PhysicsT::mNumDofsPerCell);
            mGradCurrTempFuncs.at(tName)->evaluate(tWorkSets, tResultWS);

            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumTempDofsPerNode, mNumVelDofsPerNode> tJacobianEntryOrdinal(tJacobian, &tMesh);
            Plato::assemble_transpose_jacobian(tDomain, mNumVelDofsPerCell, mNumTempDofsPerCell, tJacobianEntryOrdinal, tResultWS, tJacobian->entries());
        }

        // external force contribution
        {
            auto tNumCells = mSpatialModel.Mesh.nverts();
            Plato::Hyperbolic::FluidMechanics::WorkSets<PhysicsT, GradCurrTemp> tWorkSets(tNumCells);
            this->setWorkSets(aControls, aStates, tWorkSets);

            Plato::ScalarMultiVectorT<GradCurrTemp> tResultWS("Cells Results", tNumCells, PhysicsT::mNumDofsPerCell);
            mGradCurrTempFuncs.begin()->evaluateBoundary(tWorkSets, tResultWS);

            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumTempDofsPerNode, mNumVelDofsPerNode> tJacobianEntryOrdinal(tJacobian, &tMesh);
            Plato::assemble_transpose_jacobian(tNumCells, mNumVelDofsPerCell, mNumTempDofsPerCell, tJacobianEntryOrdinal, tResultWS, tJacobian->entries());
        }

        return tJacobian;
    }

private:
    void initialize
    (Plato::DataMap & aDataMap,
     Teuchos::ParameterList & aInputs,
     std::string & aName)
    {
        typename PhysicsT::FunctionFactory tVecFuncFactory;
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tName = tDomain.getDomainName();

            mResidualFuncs[tName]  = tVecFuncFactory.template createVectorFunction<Residual>
                (tDomain, aDataMap, aInputs, aName);

            mGradControlFuncs[tName] = tVecFuncFactory.template createVectorFunction<GradControl>
                (tDomain, aDataMap, aInputs, aName);

            mGradConfigFuncs[tName] = tVecFuncFactory.template createVectorFunction<GradConfig>
                (tDomain, aDataMap, aInputs, aName);

            mGradCurrPressFuncs[tName] = tVecFuncFactory.template createVectorFunction<GradCurrPress>
                (tDomain, aDataMap, aInputs, aName);

            mGradPrevPressFuncs[tName] = tVecFuncFactory.template createVectorFunction<GradPrevPress>
                (tDomain, aDataMap, aInputs, aName);

            mGradCurrTempFuncs[tName] = tVecFuncFactory.template createVectorFunction<GradCurrTemp>
                (tDomain, aDataMap, aInputs, aName);

            mGradPrevTempFuncs[tName] = tVecFuncFactory.template createVectorFunction<GradPrevTemp>
                (tDomain, aDataMap, aInputs, aName);

            mGradCurrVelFuncs[tName] = tVecFuncFactory.template createVectorFunction<GradCurrVel>
                (tDomain, aDataMap, aInputs, aName);

            mGradPrevVelFuncs[tName] = tVecFuncFactory.template createVectorFunction<GradPrevVel>
                (tDomain, aDataMap, aInputs, aName);

            mGradPredictorFuncs[tName] = tVecFuncFactory.template createVectorFunction<GradPredictor>
                (tDomain, aDataMap, aInputs, aName);

            mGradCorrectorFuncs[tName] = tVecFuncFactory.template createVectorFunction<GradCorrector>
                (tDomain, aDataMap, aInputs, aName);
        }
    }

    template<typename WorkSetT>
    void setWorkSets
    (const Plato::SpatialDomain & aDomain,
     const Plato::ScalarVector & aControls,
     const PrimalStates & aState,
     WorkSetT & aWorkSets)
    {
        Plato::workset_state_scalar_scalar<mNumVelDofsPerNode, mNumNodesPerCell>
            (aDomain, mVectorStateEntryOrdinal, aState.getVector("predictor"), aWorkSets.predictor());

        Plato::workset_state_scalar_scalar<mNumVelDofsPerNode, mNumNodesPerCell>
            (aDomain, mVectorStateEntryOrdinal, aState.getVector("corrector"), aWorkSets.corrector());

        Plato::workset_state_scalar_scalar<mNumVelDofsPerNode, mNumNodesPerCell>
            (aDomain, mVectorStateEntryOrdinal, aState.getVector("current velocity"), aWorkSets.currentVelocity());

        Plato::workset_state_scalar_scalar<mNumPressDofsPerNode, mNumNodesPerCell>
            (aDomain, mScalarStateEntryOrdinal, aState.getVector("current pressure"), aWorkSets.currentPressure());

        Plato::workset_state_scalar_scalar<mNumTempDofsPerNode, mNumNodesPerCell>
            (aDomain, mScalarStateEntryOrdinal, aState.getVector("current temperature"), aWorkSets.currentTemperature());

        Plato::workset_state_scalar_scalar<mNumVelDofsPerNode, mNumNodesPerCell>
            (aDomain, mVectorStateEntryOrdinal, aState.getVector("previous velocity"), aWorkSets.previousVelocity());

        Plato::workset_state_scalar_scalar<mNumPressDofsPerNode, mNumNodesPerCell>
            (aDomain, mScalarStateEntryOrdinal, aState.getVector("previous pressure"), aWorkSets.previousPressure());

        Plato::workset_state_scalar_scalar<mNumTempDofsPerNode, mNumNodesPerCell>
            (aDomain, mScalarStateEntryOrdinal, aState.getVector("previous temperature"), aWorkSets.previousTemperature());

        Plato::workset_state_scalar_scalar<mNumTimeStepsPerNode, mNumNodesPerCell>
            (aDomain, mScalarStateEntryOrdinal, aState.getVector("time step"), aWorkSets.timeStep());

        Plato::workset_control_scalar_scalar<mNumNodesPerCell>
            (aDomain, mControlEntryOrdinal, aControls, aWorkSets.control());

        Plato::workset_config_scalar<mNumSpatialDims, mNumNodesPerCell>
            (aDomain, mNodeCoordinate, aWorkSets.configuration());
    }

    template<typename WorkSetT>
    void setWorkSets
    (const Plato::ScalarVector & aControls,
     const PrimalStates & aState,
     WorkSetT & aWorkSets)
    {
        auto tNumCells = aWorkSets.numCells();
        Plato::workset_state_scalar_scalar<mNumVelDofsPerNode, mNumNodesPerCell>
            (tNumCells, mVectorStateEntryOrdinal, aState.getVector("predictor"), aWorkSets.predictor());

        Plato::workset_state_scalar_scalar<mNumVelDofsPerNode, mNumNodesPerCell>
            (tNumCells, mVectorStateEntryOrdinal, aState.getVector("corrector"), aWorkSets.corrector());

        Plato::workset_state_scalar_scalar<mNumVelDofsPerNode, mNumNodesPerCell>
            (tNumCells, mVectorStateEntryOrdinal, aState.getVector("current velocity"), aWorkSets.currentVelocity());

        Plato::workset_state_scalar_scalar<mNumPressDofsPerNode, mNumNodesPerCell>
            (tNumCells, mScalarStateEntryOrdinal, aState.getVector("current pressure"), aWorkSets.currentPressure());

        Plato::workset_state_scalar_scalar<mNumTempDofsPerNode, mNumNodesPerCell>
            (tNumCells, mScalarStateEntryOrdinal, aState.getVector("current temperature"), aWorkSets.currentTemperature());

        Plato::workset_state_scalar_scalar<mNumVelDofsPerNode, mNumNodesPerCell>
            (tNumCells, mVectorStateEntryOrdinal, aState.getVector("previous velocity"), aWorkSets.previousVelocity());

        Plato::workset_state_scalar_scalar<mNumPressDofsPerNode, mNumNodesPerCell>
            (tNumCells, mScalarStateEntryOrdinal, aState.getVector("previous pressure"), aWorkSets.previousPressure());

        Plato::workset_state_scalar_scalar<mNumTempDofsPerNode, mNumNodesPerCell>
            (tNumCells, mScalarStateEntryOrdinal, aState.getVector("previous temperature"), aWorkSets.previousTemperature());

        Plato::workset_state_scalar_scalar<mNumTimeStepsPerNode, mNumNodesPerCell>
            (tNumCells, mScalarStateEntryOrdinal, aState.getVector("time step"), aWorkSets.timeStep());

        Plato::workset_control_scalar_scalar<mNumNodesPerCell>
            (tNumCells, mControlEntryOrdinal, aControls, aWorkSets.control());

        Plato::workset_config_scalar<mNumSpatialDims, mNumNodesPerCell>
            (tNumCells, mNodeCoordinate, aWorkSets.configuration());
    }

};
// class VectorFunction

template<typename PhysicsT>
class CriterionFactory
{
private:
    using ScalarFunctionType = Plato::Hyperbolic::FluidMechanics::CriterionBase;

public:
    /******************************************************************************//**
     * \brief Constructor
     **********************************************************************************/
    CriterionFactory () {}

    /******************************************************************************//**
     * \brief Destructor
     **********************************************************************************/
    ~CriterionFactory() {}

    /******************************************************************************//**
     * \brief Creates a Plato Analyze criterion.
     * \param [in] aSpatialModel  C++ structure with volume and surface mesh databases
     * \param [in] aDataMap       Plato Analyze data map
     * \param [in] aInputs        input parameters from Analyze's input file
     * \param [in] aName          scalar function name
     **********************************************************************************/
    std::shared_ptr<ScalarFunctionType>
    createCriterion
    (Plato::SpatialModel & aSpatialModel,
     Plato::DataMap & aDataMap,
     Teuchos::ParameterList & aInputs,
     std::string & aName)
     {
        auto tFunctionTag = aInputs.sublist("Criteria").sublist(aName);
        auto tType = tFunctionTag.get<std::string>("Type", "Not Defined");
        auto tLowerType = Plato::tolower(tType);

        if(tLowerType == "scalar function")
        {
            auto tCriterion =
                std::make_shared<Plato::Hyperbolic::FluidMechanics::PhysicsScalarFunction<PhysicsT>>
                    (aSpatialModel, aDataMap, aInputs, aName);
            return tCriterion;
        }
        else
        {
            THROWERR(std::string("Scalar function in block '") + aName + "' with type '" + tType + "' is not supported.")
        }
     }
};

struct FunctionFactory
{
public:
    template <typename PhysicsT, typename EvaluationT>
    std::shared_ptr<Plato::Hyperbolic::FluidMechanics::AbstractVectorFunctionBase<PhysicsT, EvaluationT>>
    createVectorFunction
    (const Plato::SpatialDomain & aSpatialDomain,
     Plato::DataMap             & aDataMap,
     Teuchos::ParameterList     & aInputs,
     std::string                & aName)
    {
        if( !aInputs.isSublist(aName) == false )
        {
            THROWERR(std::string("Vector function with tag '") + aName + "' is not supported.")
        }

        auto tFunParams = aInputs.sublist(aName);
        auto tLowerName = Plato::tolower(aName);
        // TODO: Add pressure, velocity, temperature, predictor and corrector element residuals. explore function interface
        return nullptr;
    }

    template <typename PhysicsT, typename EvaluationT>
    std::shared_ptr<Plato::Hyperbolic::FluidMechanics::AbstractScalarFunction<PhysicsT, EvaluationT>>
    createScalarFunction
    (const Plato::SpatialDomain & aSpatialDomain,
     Plato::DataMap             & aDataMap,
     Teuchos::ParameterList     & aInputs,
     std::string                & aType,
     std::string                & aTag)
    {
        if( !aInputs.isSublist("Criteria") )
        {
            THROWERR("Criteria block is not defined.")
        }
        auto tCriteriaList = aInputs.sublist("Criteria");
        if( tCriteriaList.isSublist(aType) == false )
        {
            THROWERR(std::string("Scalar function with type '") + aType + "' is not defined.")
        }
        auto tFuncTypeList = tCriteriaList.sublist(aType);

        auto tLowerTag = Plato::tolower(aTag);
        if( tLowerTag == "pressure drop" )
        {
            // TODO: Add pressure drop and others. explore function interface
            return nullptr;
        }
        else
        {
            THROWERR(std::string("Scalar function of type '") + aType + "' and tag ' " + aTag + "' is not supported.")
        }
    }
};

}
// namespace FluidMechanics

}
// namespace Hyperbolic

template<Plato::OrdinalType SpaceDim, Plato::OrdinalType NumControls = 1>
class MomentumConservation : public Plato::SimplexFluidMechanics<SpaceDim, NumControls>
{
public:
    using SimplexT = Plato::SimplexFluidMechanics<SpaceDim, NumControls>;
    static constexpr auto mNumDofsPerNode = SimplexT::mNumMomentumDofsPerNode;
    static constexpr auto mNumDofsPerCell = SimplexT::mNumNodesPerCell * mNumDofsPerNode;
};

template<Plato::OrdinalType SpaceDim, Plato::OrdinalType NumControls = 1>
class MassConservation : public Plato::SimplexFluidMechanics<SpaceDim, NumControls>
{
public:
    using SimplexT = Plato::SimplexFluidMechanics<SpaceDim, NumControls>;
    static constexpr auto mNumDofsPerNode = SimplexT::mNumMassDofsPerNode;
    static constexpr auto mNumDofsPerCell = SimplexT::mNumNodesPerCell * mNumDofsPerNode;
};

template<Plato::OrdinalType SpaceDim, Plato::OrdinalType NumControls = 1>
class EnergyConservation : public Plato::SimplexFluidMechanics<SpaceDim, NumControls>
{
public:
    using SimplexT = Plato::SimplexFluidMechanics<SpaceDim, NumControls>;
    static constexpr auto mNumDofsPerNode = SimplexT::mNumEnergyDofsPerNode;
    static constexpr auto mNumDofsPerCell = SimplexT::mNumNodesPerCell * mNumDofsPerNode;
};

template<Plato::OrdinalType SpaceDim, Plato::OrdinalType NumControls = 1>
class IncompressibleFluids : public Plato::SimplexFluidMechanics<SpaceDim, NumControls>
{
public:
    static constexpr auto mNumSpatialDims = SpaceDim;

    typedef Plato::Hyperbolic::FluidMechanics::FunctionFactory FunctionFactory;
    using SimplexT = typename Plato::SimplexFluidMechanics<SpaceDim, NumControls>;

    using MassPhysicsT     = typename Plato::MassConservation<SpaceDim, NumControls>;
    using EnergyPhysicsT   = typename Plato::EnergyConservation<SpaceDim, NumControls>;
    using MomentumPhysicsT = typename Plato::MomentumConservation<SpaceDim, NumControls>;
};

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
(const Plato::Hyperbolic::FluidMechanics::States & aStates,
 Plato::Scalar aEpsilonConstant = 0.5)
{
    auto tPrandtl = aStates.getScalar("prandtl");
    auto tReynolds = aStates.getScalar("reynolds");
    auto tElemSize = aStates.getVector("element size");
    auto tVelocity = aStates.getVector("current velocity");

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
(const Plato::Hyperbolic::FluidMechanics::States & aStates)
{
    auto tElemSize = aStates.getVector("element size");
    auto tVelocity = aStates.getVector("current velocity");
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
(const Plato::Hyperbolic::FluidMechanics::States & aStates)
{
    auto tTimeStep = aStates.getVector("time step");
    auto tCurrentPressure = aStates.getVector("current pressure");
    auto tPreviousPressure = aStates.getVector("previous pressure");
    auto tArtificialCompress = aStates.getVector("artificial compressibility");

    auto tError = Plato::cbs::calculate_stopping_criterion(tTimeStep, tCurrentPressure, tPreviousPressure, tArtificialCompress);

    return tError;
}

inline Plato::Scalar
calculate_semi_implicit_solve_convergence_criterion
(const Plato::Hyperbolic::FluidMechanics::States & aStates)
{
    std::vector<Plato::Scalar> tErrors;

    // pressure error
    auto tTimeStep = aStates.getVector("time step");
    auto tCurrentState = aStates.getVector("current pressure");
    auto tPreviousState = aStates.getVector("previous pressure");
    auto tArtificialCompress = aStates.getVector("artificial compressibility");
    auto tMyResidual = Plato::cbs::calculate_stopping_criterion(tTimeStep, tCurrentState, tPreviousState, tArtificialCompress);
    Plato::Scalar tMyMaxStateError(0);
    Plato::blas1::max(tMyResidual, tMyMaxStateError);
    tErrors.push_back(tMyMaxStateError);

    // velocity error
    tCurrentState = aStates.getVector("current velocity");
    tPreviousState = aStates.getVector("previous velocity");
    tMyResidual = Plato::cbs::calculate_stopping_criterion(tTimeStep, tCurrentState, tPreviousState, tArtificialCompress);
    tMyMaxStateError(0.0);
    Plato::blas1::max(tMyResidual, tMyMaxStateError);
    tErrors.push_back(tMyMaxStateError);

    // temperature error
    tCurrentState = aStates.getVector("current temperature");
    tPreviousState = aStates.getVector("previous temperature");
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

    virtual void output(std::string aFilePath) = 0;
    virtual const Plato::DataMap& getDataMap() const = 0;
    virtual Plato::Hyperbolic::Solution solution(const Plato::ScalarVector& aControl) = 0;
    virtual Plato::Scalar criterionValue(const Plato::ScalarVector & aControl, const std::string& aName) = 0;
    virtual Plato::ScalarVector criterionGradient(const Plato::ScalarVector &aControl, const std::string &aName) = 0;
    virtual Plato::ScalarVector criterionGradientX(const Plato::ScalarVector &aControl, const std::string &aName) = 0;
};

template<typename PhysicsT>
class FluidMechanicsProblem : public Plato::Hyperbolic::AbstractProblem
{
private:
    static constexpr auto mNumSpatialDims     = PhysicsT::mNumSpatialDims;         /*!< number of mass dofs per node */
    static constexpr auto mNumVelDofsPerNode  = PhysicsT::mNumMomentumDofsPerNode; /*!< number of momentum dofs per node */
    static constexpr auto mNumPresDofsPerNode = PhysicsT::mNumMassDofsPerNode;     /*!< number of mass dofs per node */
    static constexpr auto mNumTempDofsPerNode = PhysicsT::mNumEnergyDofsPerNode;   /*!< number of energy dofs per node */

    Plato::DataMap      mDataMap;
    Plato::SpatialModel mSpatialModel; /*!< SpatialModel instance contains the mesh, meshsets, domains, etc. */

    bool mIsExplicitSolve = true;
    bool mIsTransientProblem = false;

    Plato::Scalar mPrandtlNumber = 1.0;
    Plato::Scalar mReynoldsNumber = 1.0;
    Plato::Scalar mCBSsolverTolerance = 1e-5;
    Plato::OrdinalType mNumTimeSteps = 100;

    Plato::ScalarMultiVector mPressure;
    Plato::ScalarMultiVector mVelocity;
    Plato::ScalarMultiVector mPredictor;
    Plato::ScalarMultiVector mCorrector;
    Plato::ScalarMultiVector mTemperature;

    Plato::Hyperbolic::FluidMechanics::VectorFunction<typename PhysicsT::MassPhysicsT>     mPressureResidual;
    Plato::Hyperbolic::FluidMechanics::VectorFunction<typename PhysicsT::MomentumPhysicsT> mVelocityResidual;
    Plato::Hyperbolic::FluidMechanics::VectorFunction<typename PhysicsT::MomentumPhysicsT> mPredictorResidual;
    Plato::Hyperbolic::FluidMechanics::VectorFunction<typename PhysicsT::MomentumPhysicsT> mCorrectorResidual;
    Plato::Hyperbolic::FluidMechanics::VectorFunction<typename PhysicsT::EnergyPhysicsT>   mTemperatureResidual;

    using Criterion = std::shared_ptr<Plato::Hyperbolic::FluidMechanics::CriterionBase>;
    using Criteria  = std::unordered_map<std::string, Criterion>;
    Criteria mCriteria;

    using MassConservationT     = typename Plato::MassConservation<PhysicsT::mNumSpatialDims, PhysicsT::mNumControls>;
    using EnergyConservationT   = typename Plato::EnergyConservation<PhysicsT::mNumSpatialDims, PhysicsT::mNumControls>;
    using MomentumConservationT = typename Plato::MomentumConservation<PhysicsT::mNumSpatialDims, PhysicsT::mNumControls>;
    Plato::EssentialBCs<MassConservationT>     mPressureStateBoundaryConditions;
    Plato::EssentialBCs<MomentumConservationT> mVelocityStateBoundaryConditions;
    Plato::EssentialBCs<EnergyConservationT>   mTemperatureStateBoundaryConditions;

    std::shared_ptr<Plato::AbstractSolver> mVectorFieldSolver;
    std::shared_ptr<Plato::AbstractSolver> mScalarFieldSolver;

    using DualStates = Plato::Hyperbolic::FluidMechanics::Dual;
    using PrimalStates = Plato::Hyperbolic::FluidMechanics::States;

public:
    FluidMechanicsProblem
    (Omega_h::Mesh & aMesh,
     Omega_h::MeshSets & aMeshSets,
     Teuchos::ParameterList & aInputs,
     Comm::Machine & aMachine) :
         mSpatialModel       (aMesh, aMeshSets, aInputs),
         mPressureResidual   (mSpatialModel, mDataMap, aInputs, "Pressure Residual"),
         mVelocityResidual   (mSpatialModel, mDataMap, aInputs, "Velocity Residual"),
         mPredictorResidual  (mSpatialModel, mDataMap, aInputs, "Predictor Residual"),
         mCorrectorResidual  (mSpatialModel, mDataMap, aInputs, "Corrector Residual"),
         mTemperatureResidual(mSpatialModel, mDataMap, aInputs, "Temperature Residual")
    {
        this->initialize(aInputs, aMachine);
    }

    const decltype(mDataMap)& getDataMap() const
    {
        return mDataMap;
    }

    void output(std::string aFilePath = "output")
    {
        auto tMesh = mSpatialModel.Mesh;
        const auto tTimeSteps = mVelocity.extent(0);
        auto tWriter = Omega_h::vtk::Writer(aFilePath.c_str(), &tMesh, mNumSpatialDims);

        constexpr auto tStride = 0;
        const auto tNumNodes = tMesh.nverts();
        for(decltype(tTimeSteps) tStep = 0; tStep < tTimeSteps; tStep++)
        {
            auto tPressSubView = Kokkos::subview(mPressure, tStep, Kokkos::ALL());
            Omega_h::Write<Omega_h::Real> tPressure(tPressSubView.size(), "Pressure");
            Plato::copy<mNumPresDofsPerNode, mNumPresDofsPerNode>(tStride, tNumNodes, tPressSubView, tPressure);
            tMesh.add_tag(Omega_h::VERT, "Pressure", mNumPresDofsPerNode, Omega_h::Reals(tPressure));

            auto tTempSubView = Kokkos::subview(mTemperature, tStep, Kokkos::ALL());
            Omega_h::Write<Omega_h::Real> tTemperature(tTempSubView.size(), "Temperature");
            Plato::copy<mNumTempDofsPerNode, mNumTempDofsPerNode>(tStride, tNumNodes, tTempSubView, tTemperature);
            tMesh.add_tag(Omega_h::VERT, "Temperature", mNumTempDofsPerNode, Omega_h::Reals(tTemperature));

            auto tVelSubView = Kokkos::subview(mVelocity, tStep, Kokkos::ALL());
            Omega_h::Write<Omega_h::Real> tVelocity(tVelSubView.size(), "Velocity");
            Plato::copy<mNumVelDofsPerNode, mNumVelDofsPerNode>(tStride, tNumNodes, tVelSubView, tVelocity);
            tMesh.add_tag(Omega_h::VERT, "Velocity", mNumVelDofsPerNode, Omega_h::Reals(tVelocity));

            auto tTags = Omega_h::vtk::get_all_vtk_tags(&tMesh, mNumSpatialDims);
            auto tTime = static_cast<Plato::Scalar>(1.0 / tTimeSteps) * static_cast<Plato::Scalar>(tStep + 1);
            tWriter.write(tStep, tTime, tTags);
        }
    }

    Plato::Hyperbolic::Solution solution(const Plato::ScalarVector& aControl)
    {
        PrimalStates tStates;
        this->calculateElemCharacteristicSize(tStates);

        for (decltype(mNumTimeSteps) tStep = 1; tStep < mNumTimeSteps; tStep++)
        {
            tStates.setScalar("step", tStep);
            this->setPrimalStates(tStates);
            this->calculateStableTimeSteps(tStates);

            this->calculateVelocityPredictor(aControl, tStates);
            this->calculatePressureState(aControl, tStates);
            this->calculateVelocityCorrector(aControl, tStates);
            this->calculateTemperatureState(aControl, tStates);

            this->enforceVelocityBoundaryConditions(tStates);
            this->enforcePressureBoundaryConditions(tStates);
            this->enforceTemperatureBoundaryConditions(tStates);

            if(this->checkStoppingCriteria(tStates))
            {
                break;
            }
        }

        Plato::Hyperbolic::Solution tSolution;
        tSolution.set("mass state", mPressure);
        tSolution.set("energy state", mTemperature);
        tSolution.set("momentum state", mVelocity);
        return tSolution;
    }

    Plato::Scalar criterionValue
    (const Plato::ScalarVector & aControl,
     const std::string & aName)
    {
        auto tItr = mCriteria.find(aName);
        if (tItr == mCriteria.end())
        {
            PrimalStates tPrimalStates;
            this->calculateElemCharacteristicSize(tPrimalStates);

            Plato::Scalar tOutput(0);
            auto tNumTimeSteps = mVelocity.extent(0);
            for (decltype(tNumTimeSteps) tStep = 0; tStep < tNumTimeSteps; tStep++)
            {
                tPrimalStates.setScalar("step", tStep);
                auto tPressure = Kokkos::subview(mPressure, tStep, Kokkos::ALL());
                auto tVelocity = Kokkos::subview(mVelocity, tStep, Kokkos::ALL());
                auto tTemperature = Kokkos::subview(mTemperature, tStep, Kokkos::ALL());
                tPrimalStates.setVector("current pressure", tPressure);
                tPrimalStates.setVector("current velocity", tVelocity);
                tPrimalStates.setVector("current temperature", tTemperature);
                tOutput += tItr->second->value(aControl, tPrimalStates);
            }
            return tOutput;
        }
        else
        {
            THROWERR(std::string("Criterion with tag '") + aName + "' is not supported");
        }
    }

    Plato::ScalarVector criterionGradient
    (const Plato::ScalarVector & aControl,
     const std::string & aName)
    {
        auto tItr = mCriteria.find(aName);
        if (tItr == mCriteria.end())
        {
            DualStates   tDualStates;
            PrimalStates tCurrentStates, tPreviousStates;
            this->calculateElemCharacteristicSize(tCurrentStates);

            auto tLastStepIndex = mVelocity.extent(0) - 1;
            Plato::ScalarVector tTotalDerivative("Total Derivative", aControl.size());
            for(decltype(tLastStepIndex) tStep = tLastStepIndex; tStep >= 0; tStep--)
            {
                tDualStates.setScalar("step", tStep);
                tCurrentStates.setScalar("step", tStep);
                tPreviousStates.setScalar("step", tStep + 1);

                this->setDualStates(tDualStates);
                this->setPrimalStates(tCurrentStates);
                this->setPrimalStates(tPreviousStates);

                this->calculateStableTimeSteps(tCurrentStates);
                this->calculateStableTimeSteps(tPreviousStates);

                this->calculateMomentumAdjoint(aName, aControl, tCurrentStates, tPreviousStates, tDualStates);
                this->calculateTemperatureAdjoint(aName, aControl, tCurrentStates, tPreviousStates, tDualStates);
                this->calculateCorrectorAdjoint(aControl, tCurrentStates, tDualStates);
                this->calculatePressureAdjoint(aName, aControl, tCurrentStates, tPreviousStates, tDualStates);
                this->calculatePredictorAdjoint(aControl, tCurrentStates, tDualStates);

                this->calculateGradientControl(aName, aControl, tCurrentStates, tDualStates, tTotalDerivative);
            }

            return tTotalDerivative;
        }
        else
        {
            THROWERR(std::string("Criterion with tag '") + aName + "' is not supported");
        }
    }

    Plato::ScalarVector criterionGradientX
    (const Plato::ScalarVector & aControl,
     const std::string & aName)
    {
        auto tItr = mCriteria.find(aName);
        if (tItr == mCriteria.end())
        {
            DualStates   tDualStates;
            PrimalStates tCurrentStates, tPreviousStates;
            this->calculateElemCharacteristicSize(tCurrentStates);

            auto tLastStepIndex = mVelocity.extent(0) - 1;
            Plato::ScalarVector tTotalDerivative("Total Derivative", aControl.size());
            for(decltype(tLastStepIndex) tStep = tLastStepIndex; tStep >= 0; tStep--)
            {
                tDualStates.setScalar("step", tStep);
                tCurrentStates.setScalar("step", tStep);
                tPreviousStates.setScalar("step", tStep + 1);

                this->setDualStates(tDualStates);
                this->setPrimalStates(tCurrentStates);
                this->setPrimalStates(tPreviousStates);

                this->calculateStableTimeSteps(tCurrentStates);
                this->calculateStableTimeSteps(tPreviousStates);

                this->calculateMomentumAdjoint(aName, aControl, tCurrentStates, tPreviousStates, tDualStates);
                this->calculateTemperatureAdjoint(aName, aControl, tCurrentStates, tPreviousStates, tDualStates);
                this->calculateCorrectorAdjoint(aControl, tCurrentStates, tDualStates);
                this->calculatePressureAdjoint(aName, aControl, tCurrentStates, tPreviousStates, tDualStates);
                this->calculatePredictorAdjoint(aControl, tCurrentStates, tDualStates);

                this->calculateGradientConfig(aName, aControl, tCurrentStates, tDualStates, tTotalDerivative);
            }

            return tTotalDerivative;
        }
        else
        {
            THROWERR(std::string("Criterion with tag '") + aName + "' is not supported");
        }
    }

private:
    void initialize
    (Teuchos::ParameterList &aInputs,
     Comm::Machine& aMachine)
    {
        Plato::SolverFactory tSolverFactory(aInputs.sublist("Linear Solver"));
        mVectorFieldSolver = tSolverFactory.create(mSpatialModel.Mesh, aMachine, mNumVelDofsPerNode);
        mScalarFieldSolver = tSolverFactory.create(mSpatialModel.Mesh, aMachine, mNumPresDofsPerNode);

        if(aInputs.isSublist("Time Integration"))
        {
            mNumTimeSteps = aInputs.sublist("Time Integration").get<int>("Number Time Steps", 100);
        }
        auto tNumNodes = mSpatialModel.Mesh.nverts();
        mPressure    = Plato::ScalarMultiVector("Pressure Snapshots", mNumTimeSteps, tNumNodes);
        mVelocity    = Plato::ScalarMultiVector("Velocity Snapshots", mNumTimeSteps, tNumNodes * mNumVelDofsPerNode);
        mPredictor   = Plato::ScalarMultiVector("Velocity Snapshots", mNumTimeSteps, tNumNodes * mNumVelDofsPerNode);
        mCorrector   = Plato::ScalarMultiVector("Velocity Snapshots", mNumTimeSteps, tNumNodes * mNumVelDofsPerNode);
        mTemperature = Plato::ScalarMultiVector("Temperature Snapshots", mNumTimeSteps, tNumNodes);

        this->parseCriteria(aInputs);
        this->readBoundaryConditions(aInputs);
    }

    void readBoundaryConditions(Teuchos::ParameterList& aInputs)
    {
        auto tReadBCs = aInputs.get<bool>("Read Boundary Conditions", true);
        if (tReadBCs)
        {
            this->readPressureBoundaryConditions(aInputs);
            this->readVelocityBoundaryConditions(aInputs);
            this->readTemperatureBoundaryConditions(aInputs);
        }
    }

    void parseCriteria(Teuchos::ParameterList &aInputs)
    {
        if(aInputs.isSublist("Criteria"))
        {
            Plato::Hyperbolic::FluidMechanics::CriterionFactory<PhysicsT> tScalarFuncFactory;

            auto tCriteriaParams = aInputs.sublist("Criteria");
            for(Teuchos::ParameterList::ConstIterator tIndex = tCriteriaParams.begin(); tIndex != tCriteriaParams.end(); ++tIndex)
            {
                const Teuchos::ParameterEntry& tEntry = tCriteriaParams.entry(tIndex);
                if(tEntry.isList())
                {
                    THROWERR("Parameter in Criteria block is not supported.  Expect lists only.")
                }
                auto tName = tCriteriaParams.name(tIndex);
                auto tCriterion = tScalarFuncFactory.createCriterion(mSpatialModel, mDataMap, aInputs, tName);
                if( tCriterion != nullptr )
                {
                    mCriteria[tName] = tCriterion;
                }
            }
        }
    }

    bool checkStoppingCriteria(const PrimalStates & aStates)
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

            Plato::OrdinalType tStep = aStates.getScalar("step");
            if(tCriterionValue < mCBSsolverTolerance)
            {
                tStop = true;
            }
            else if(tStep >= mNumTimeSteps)
            {
                tStop = true;
            }
        }
        return tStop;
    }

    void calculateElemCharacteristicSize(PrimalStates & aStates)
    {
        Plato::ScalarVector tElementSize;
        Plato::NodeCoordinate<mNumSpatialDims> tNodeCoordinates;
        auto tNumCells = mSpatialModel.Mesh.nverts();
        auto tElemCharacteristicSize =
            Plato::cbs::calculate_element_characteristic_size<mNumSpatialDims>(tNumCells, tNodeCoordinates);
        aStates.setVector("element size", tElemCharacteristicSize);
    }

    void calculateStableTimeSteps(PrimalStates & aStates)
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

    void enforceVelocityBoundaryConditions(PrimalStates & aStates)
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
        auto tCurrentVelocity = aStates.getVector("current velocity");
        Plato::cbs::enforce_boundary_condition(tBcDofs, tBcValues, tCurrentVelocity);
    }

    void enforcePressureBoundaryConditions(PrimalStates& aStates)
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
        auto tCurrentPressure = aStates.getVector("current pressure");
        Plato::cbs::enforce_boundary_condition(tBcDofs, tBcValues, tCurrentPressure);
    }

    void enforceTemperatureBoundaryConditions(PrimalStates & aStates)
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
        auto tCurrentTemperature = aStates.getVector("current temperature");
        Plato::cbs::enforce_boundary_condition(tBcDofs, tBcValues, tCurrentTemperature);
    }

    void setDualStates(DualStates & aStates)
    {
        if(aStates.isVectorMapEmpty())
        {
            // FIRST BACKWARD TIME INTEGRATION STEP
            auto tTotalNumNodes = mSpatialModel.Mesh.nverts();
            std::vector<std::string> tNames = {"current pressure adjoint" , "current temperature adjoint",
                "previous pressure adjoint", "previous temperature adjoint"};
            for(auto& tName : tNames)
            {
                Plato::ScalarVector tView(tName, tTotalNumNodes);
                aStates.setVector(tName, tView);
            }

            auto tTotalNumDofs = mNumVelDofsPerNode * tTotalNumNodes;
            tNames = {"current velocity adjoint" , "current predictor adjoint" , "current corrector adjoint",
                "previous velocity adjoint", "previous predictor adjoint", "previous corrector adjoint"};
            for(auto& tName : tNames)
            {
                Plato::ScalarVector tView(tName, tTotalNumDofs);
                aStates.setVector(tName, tView);
            }
        }
        else
        {
            // N-TH BACKWARD TIME INTEGRATION STEP
            std::vector<std::string> tNames = {"mass adjoint", "energy adjoint",
                "momentum adjoint", "predictor adjoint" , "corrector adjoint"};
            for(auto& tName : tNames)
            {
                auto tVector = aStates.getVector(std::string("current ") + tName);
                aStates.setVector(std::string("previous ") + tName, tVector);
            }
        }
    }

    void setPrimalStates(PrimalStates & aStates)
    {
        Plato::OrdinalType tStep = aStates.getScalar("step");
        auto tCurrentMass       = Kokkos::subview(mPressure, tStep, Kokkos::ALL());
        auto tCurrentEnergy     = Kokkos::subview(mTemperature, tStep, Kokkos::ALL());
        auto tCurrentMomentum   = Kokkos::subview(mVelocity, tStep, Kokkos::ALL());
        auto tMomentumPredictor = Kokkos::subview(mPredictor, tStep, Kokkos::ALL());
        auto tMomentumCorrector = Kokkos::subview(mCorrector, tStep, Kokkos::ALL());
        aStates.setVector("current pressure", tCurrentMass);
        aStates.setVector("current temperature", tCurrentEnergy);
        aStates.setVector("current velocity", tCurrentMomentum);
        aStates.setVector("predictor", tMomentumPredictor);
        aStates.setVector("corrector", tMomentumPredictor);

        auto tPrevStep = tStep - 1;
        if (tPrevStep >= static_cast<Plato::OrdinalType>(0))
        {
            auto tPreviousMass     = Kokkos::subview(mPressure, tPrevStep, Kokkos::ALL());
            auto tPreviousEnergy   = Kokkos::subview(mTemperature, tPrevStep, Kokkos::ALL());
            auto tPreviousMomentum = Kokkos::subview(mVelocity, tPrevStep, Kokkos::ALL());
            aStates.setVector("previous pressure", tPreviousMass);
            aStates.setVector("previous temperature", tPreviousEnergy);
            aStates.setVector("previous velocity", tPreviousMomentum);
        }
        else
        {
            auto tLength = mPressure.extent(1);
            Plato::ScalarVector tPreviousMass("previous pressure", tLength);
            aStates.setVector("previous pressure", tPreviousMass);
            tLength = mTemperature.extent(1);
            Plato::ScalarVector tPreviousEnergy("previous temperature", tLength);
            aStates.setVector("previous temperature", tPreviousEnergy);
            tLength = mVelocity.extent(1);
            Plato::ScalarVector tPreviousMomentum("previous velocity", tLength);
            aStates.setVector("previous velocity", tPreviousMomentum);
        }
    }

    void calculateVelocityCorrector
    (const Plato::ScalarVector& aControl,
     PrimalStates & aStates)
    {
        aStates.function("vector function");

        // calculate current residual and jacobian matrix
        auto tResidualCorrector = mCorrectorResidual.value(aStates);
        auto tJacobianCorrector = mCorrectorResidual.gradientCorrector(aStates);

        // solve corrector equation (consistent or mass lumped)
        auto tVelocityCorrector = aStates.getVector("corrector");
        Plato::blas1::fill(static_cast<Plato::Scalar>(0.0), tVelocityCorrector);
        mScalarFieldSolver->solve(*tJacobianCorrector, tVelocityCorrector, tResidualCorrector);

        // update velocity
        auto tCurrentVelocity = aStates.getVector("current velocity");
        auto tPreviousVelocity = aStates.getVector("previous velocity");
        auto tVelocityPredictor = aStates.getVector("predictor");
        Plato::blas1::copy(tPreviousVelocity, tCurrentVelocity);
        Plato::blas1::axpy(1.0, tVelocityPredictor, tCurrentVelocity);
        Plato::blas1::axpy(1.0, tVelocityCorrector, tCurrentVelocity);
    }

    void calculateVelocityPredictor
    (const Plato::ScalarVector& aControl,
     PrimalStates & aStates)
    {
        aStates.function("vector function");

        // calculate current residual and jacobian matrix
        auto tResidualPredictor = mPredictorResidual.value(aStates);
        auto tJacobianPredictor = mPredictorResidual.gradientPredictor(aStates);

        // solve predictor equation (consistent or mass lumped)
        auto tVelocityPredictor = aStates.getVector("predictor");
        Plato::blas1::fill(static_cast<Plato::Scalar>(0.0), tVelocityPredictor);
        mVectorFieldSolver->solve(*tJacobianPredictor, tVelocityPredictor, tResidualPredictor);
    }

    void calculatePressureState
    (const Plato::ScalarVector& aControl,
     PrimalStates & aStates)
    {
        aStates.function("scalar function");

        // calculate current residual and jacobian matrix
        auto tResidualPressure = mPressureResidual.value(aStates);
        auto tJacobianPressure = mPressureResidual.gradientCurrMass(aStates);

        // solve mass equation (consistent or mass lumped)
        Plato::ScalarVector tDeltaPressure("increment", tResidualPressure.size());
        Plato::blas1::fill(static_cast<Plato::Scalar>(0.0), tDeltaPressure);
        mScalarFieldSolver->solve(*tJacobianPressure, tDeltaPressure, tResidualPressure);

        // update pressure
        auto tCurrentPressure = aStates.getVector("current pressure");
        auto tPreviousPressure = aStates.getVector("previous pressure");
        Plato::blas1::copy(tPreviousPressure, tCurrentPressure);
        Plato::blas1::axpy(1.0, tDeltaPressure, tCurrentPressure);
    }

    void calculateTemperatureState
    (const Plato::ScalarVector& aControl,
     PrimalStates & aStates)
    {
        aStates.function("scalar function");

        // calculate current residual and jacobian matrix
        auto tResidualTemperature = mTemperatureResidual.value(aStates);
        auto tJacobianTemperature = mTemperatureResidual.gradientCurrEnergy(aStates);

        // solve energy equation (consistent or mass lumped)
        Plato::ScalarVector tDeltaTemperature("increment", tResidualTemperature.size());
        Plato::blas1::fill(static_cast<Plato::Scalar>(0.0), tDeltaTemperature);
        mScalarFieldSolver->solve(*tJacobianTemperature, tDeltaTemperature, tResidualTemperature);

        // update temperature
        auto tCurrentTemperature = aStates.getVector("current temperature");
        auto tPreviousTemperature = aStates.getVector("previous temperature");
        Plato::blas1::copy(tPreviousTemperature, tCurrentTemperature);
        Plato::blas1::axpy(1.0, tDeltaTemperature, tCurrentTemperature);
    }

    void calculatePredictorAdjoint
    (const Plato::ScalarVector & aControl,
     const PrimalStates & aCurrentStates,
     DualStates & aDualStates)
    {

        auto tCurrentVelocityAdjoint = aDualStates.getVector("current velocity adjoint");
        auto tGradResVelWrtPredictor = mVelocityResidual.gradientPredictor(aControl, aCurrentStates);
        Plato::ScalarVector tRHS("right hand side", tCurrentVelocityAdjoint.size());
        Plato::MatrixTimesVectorPlusVector(tGradResVelWrtPredictor, tCurrentVelocityAdjoint, tRHS);

        auto tCurrentPressureAdjoint = aDualStates.getVector("current pressure adjoint");
        auto tGradResPressWrtPredictor = mPressureResidual.gradientPredictor(aControl, aCurrentStates);
        Plato::MatrixTimesVectorPlusVector(tGradResPressWrtPredictor, tCurrentPressureAdjoint, tRHS);
        Plato::blas1::scale(-1.0, tRHS);

        auto tCurrentPredictorAdjoint = aDualStates.getVector("current predictor adjoint");
        Plato::blas1::fill(0.0, tCurrentPredictorAdjoint);
        auto tJacobianPredictor = mPredictorResidual.gradientPredictor(aControl, aCurrentStates);
        mVectorFieldSolver->solve(*tJacobianPredictor, tCurrentPredictorAdjoint, tRHS);
    }

    void calculateCorrectorAdjoint
    (const Plato::ScalarVector & aControl,
     const PrimalStates & aCurrentStates,
     DualStates & aDualStates)
    {
        auto tCurrentVelocityAdjoint = aDualStates.getVector("current velocity adjoint");
        auto tGradResVelWrtCorrector = mVelocityResidual.gradientCorrector(aControl, aCurrentStates);
        Plato::ScalarVector tRHS("right hand side", tCurrentVelocityAdjoint.size());
        Plato::MatrixTimesVectorPlusVector(tGradResVelWrtCorrector, tCurrentVelocityAdjoint, tRHS);
        Plato::blas1::scale(-1.0, tRHS);

        auto tCurrentCorrectorAdjoint = aDualStates.getVector("current corrector adjoint");
        Plato::blas1::fill(0.0, tCurrentCorrectorAdjoint);
        auto tJacobianCorrector = mCorrectorResidual.gradientCorrector(aControl, aCurrentStates);
        mVectorFieldSolver->solve(*tJacobianCorrector, tCurrentCorrectorAdjoint, tRHS);
    }

    void calculatePressureAdjoint
    (const std::string & aName,
     const Plato::ScalarVector & aControl,
     const PrimalStates & aCurrentStates,
     const PrimalStates & aPreviousStates,
     DualStates & aDualStates)
    {
        auto tRHS = mCriteria[aName]->gradientCurrentPress(aControl, aCurrentStates);

        auto tGradResCorrWrtCurPres = mCorrectorResidual.gradientCurrentPress(aControl, aCurrentStates);
        auto tCurrentCorrectorAdjoint = aDualStates.getVector("current corrector adjoint");
        Plato::MatrixTimesVectorPlusVector(tGradResCorrWrtCurPres, tCurrentCorrectorAdjoint, tRHS);

        auto tGradResPressWrtPrevPress = mPressureResidual.gradientPreviousPress(aControl, aPreviousStates);
        auto tPrevPressureAdjoint = aDualStates.getVector("previous pressure adjoint");
        Plato::MatrixTimesVectorPlusVector(tGradResPressWrtPrevPress, tPrevPressureAdjoint, tRHS);

        auto tGradResCorrWrtPrevPress = mCorrectorResidual.gradientPreviousPress(aControl, aPreviousStates);
        auto tPrevCorrectorAdjoint = aDualStates.getVector("previous corrector adjoint");
        Plato::MatrixTimesVectorPlusVector(tGradResCorrWrtPrevPress, tPrevCorrectorAdjoint, tRHS);
        Plato::blas1::scale(-1.0, tRHS);

        auto tCurrentPressAdjoint = aDualStates.getVector("current pressure adjoint");
        Plato::blas1::fill(0.0, tCurrentPressAdjoint);
        auto tJacobianPress = mPressureResidual.gradientCurrentPress(aControl, aCurrentStates);
        mVectorFieldSolver->solve(*tJacobianPress, tCurrentPressAdjoint, tRHS);
    }

    void calculateTemperatureAdjoint
    (const std::string & aName,
     const Plato::ScalarVector & aControl,
     const PrimalStates & aCurrentStates,
     const PrimalStates & aPreviousStates,
     DualStates & aDualStates)
    {
        auto tRHS = mCriteria[aName]->gradientCurrentTemp(aControl, aCurrentStates);

        auto tGradResPredWrtPrevTemp = mPredictorResidual.gradientPreviousTemp(aControl, aPreviousStates);
        auto tPrevPredictorAdjoint = aDualStates.getVector("previous predictor adjoint");
        Plato::MatrixTimesVectorPlusVector(tGradResPredWrtPrevTemp, tPrevPredictorAdjoint, tRHS);

        auto tGradResTempWrtPrevTemp = mTemperatureResidual.gradientPreviousTemp(aControl, aPreviousStates);
        auto tPrevTempAdjoint = aDualStates.getVector("previous temperature adjoint");
        Plato::MatrixTimesVectorPlusVector(tGradResTempWrtPrevTemp, tPrevTempAdjoint, tRHS);
        Plato::blas1::scale(-1.0, tRHS);

        auto tCurrentTempAdjoint = aDualStates.getVector("current temperature adjoint");
        Plato::blas1::fill(0.0, tCurrentTempAdjoint);
        auto tJacobianTemp = mTemperatureResidual.gradientCurrentTemp(aControl, aCurrentStates);
        mVectorFieldSolver->solve(*tJacobianTemp, tCurrentTempAdjoint, tRHS);
    }

    void calculateMomentumAdjoint
    (const std::string & aName,
     const Plato::ScalarVector & aControl,
     const PrimalStates & aCurrentStates,
     const PrimalStates & aPreviousStates,
     DualStates & aDualStates)
    {
        auto tRHS = mCriteria[aName]->gradientCurrentVel(aControl, aCurrentStates);

        auto tGradResPredWrtPrevVel = mPredictorResidual.gradientPreviousVel(aControl, aPreviousStates);
        auto tPrevPredictorAdjoint = aDualStates.getVector("previous predictor adjoint");
        Plato::MatrixTimesVectorPlusVector(tGradResPredWrtPrevVel, tPrevPredictorAdjoint, tRHS);

        auto tGradResCorrWrtPrevVel = mCorrectorResidual.gradientPreviousVel(aControl, aPreviousStates);
        auto tPrevCorrectorAdjoint = aDualStates.getVector("previous corrector adjoint");
        Plato::MatrixTimesVectorPlusVector(tGradResCorrWrtPrevVel, tPrevCorrectorAdjoint, tRHS);

        auto tGradResPressWrtPrevVel = mPressureResidual.gradientPreviousVel(aControl, aPreviousStates);
        auto tPrevPressureAdjoint = aDualStates.getVector("previous pressure adjoint");
        Plato::MatrixTimesVectorPlusVector(tGradResPressWrtPrevVel, tPrevPressureAdjoint, tRHS);

        auto tGradResTempWrtPrevVel = mTemperatureResidual.gradientPreviousVel(aControl, aPreviousStates);
        auto tPrevTemperatureAdjoint = aDualStates.getVector("previous temperature adjoint");
        Plato::MatrixTimesVectorPlusVector(tGradResTempWrtPrevVel, tPrevTemperatureAdjoint, tRHS);

        auto tPrevVelocityAdjoint = aDualStates.getVector("previous velocity adjoint");
        auto tGradResVelWrtPrevVel = mVelocityResidual.gradientPreviousVel(aControl, aPreviousStates);
        Plato::MatrixTimesVectorPlusVector(tGradResVelWrtPrevVel, tPrevVelocityAdjoint, tRHS);
        Plato::blas1::scale(-1.0, tRHS);

        auto tCurrentVelocityAdjoint = aDualStates.getVector("current velocity adjoint");
        Plato::blas1::fill(0.0, tCurrentVelocityAdjoint);
        auto tJacobianVel = mVelocityResidual.gradientCurrentVel(aControl, aCurrentStates);
        mVectorFieldSolver->solve(*tJacobianVel, tCurrentVelocityAdjoint, tRHS);
    }

    void calculateGradientControl
    (const std::string & aName,
     const Plato::ScalarVector & aControl,
     const PrimalStates & aCurrentStates,
     const DualStates & aDualStates,
     Plato::ScalarVector & aTotalDerivative)
    {
        auto tGradCriterionWrtControl = mCriteria[aName]->gradientControl(aControl, aCurrentStates);

        auto tCurrentPredictorAdjoint = aDualStates.getVector("current predictor adjoint");
        auto tGradResPredWrtControl = mPredictorResidual.gradientControl(aControl, aCurrentStates);
        Plato::MatrixTimesVectorPlusVector(tGradResPredWrtControl, tCurrentPredictorAdjoint, tGradCriterionWrtControl);

        auto tCurrentPressureAdjoint = aDualStates.getVector("current pressure adjoint");
        auto tGradResPressWrtControl = mPressureResidual.gradientControl(aControl, aCurrentStates);
        Plato::MatrixTimesVectorPlusVector(tGradResPressWrtControl, tCurrentPressureAdjoint, tGradCriterionWrtControl);

        auto tCurrentCorrectorAdjoint = aDualStates.getVector("current corrector adjoint");
        auto tGradResCorrWrtControl = mCorrectorResidual.gradientControl(aControl, aCurrentStates);
        Plato::MatrixTimesVectorPlusVector(tGradResCorrWrtControl, tCurrentCorrectorAdjoint, tGradCriterionWrtControl);

        auto tCurrentTemperatureAdjoint = aDualStates.getVector("current temperature adjoint");
        auto tGradResTempWrtControl = mTemperatureResidual.gradientControl(aControl, aCurrentStates);
        Plato::MatrixTimesVectorPlusVector(tGradResTempWrtControl, tCurrentTemperatureAdjoint, tGradCriterionWrtControl);

        auto tCurrentVelocityAdjoint = aDualStates.getVector("current velocity adjoint");
        auto tGradResVelWrtControl = mVelocityResidual.gradientControl(aControl, aCurrentStates);
        Plato::MatrixTimesVectorPlusVector(tGradResVelWrtControl, tCurrentVelocityAdjoint, tGradCriterionWrtControl);

        Plato::blas1::axpy(1.0, tGradCriterionWrtControl, aTotalDerivative);
    }

    void calculateGradientConfig
    (const std::string & aName,
     const Plato::ScalarVector & aControl,
     const PrimalStates & aCurrentStates,
     const DualStates & aDualStates,
     Plato::ScalarVector & aTotalDerivative)
    {
        auto tGradCriterionWrtConfig = mCriteria[aName]->gradientConfig(aControl, aCurrentStates);

        auto tCurrentPredictorAdjoint = aDualStates.getVector("current predictor adjoint");
        auto tGradResPredWrtConfig = mPredictorResidual.gradientConfig(aControl, aCurrentStates);
        Plato::MatrixTimesVectorPlusVector(tGradResPredWrtConfig, tCurrentPredictorAdjoint, tGradCriterionWrtConfig);

        auto tCurrentPressureAdjoint = aDualStates.getVector("current pressure adjoint");
        auto tGradResPressWrtConfig = mPressureResidual.gradientConfig(aControl, aCurrentStates);
        Plato::MatrixTimesVectorPlusVector(tGradResPressWrtConfig, tCurrentPressureAdjoint, tGradCriterionWrtConfig);

        auto tCurrentCorrectorAdjoint = aDualStates.getVector("current corrector adjoint");
        auto tGradResCorrWrtConfig = mCorrectorResidual.gradientConfig(aControl, aCurrentStates);
        Plato::MatrixTimesVectorPlusVector(tGradResCorrWrtConfig, tCurrentCorrectorAdjoint, tGradCriterionWrtConfig);

        auto tCurrentTemperatureAdjoint = aDualStates.getVector("current temperature adjoint");
        auto tGradResTempWrtConfig = mTemperatureResidual.gradientConfig(aControl, aCurrentStates);
        Plato::MatrixTimesVectorPlusVector(tGradResTempWrtConfig, tCurrentTemperatureAdjoint, tGradCriterionWrtConfig);

        auto tCurrentVelocityAdjoint = aDualStates.getVector("current velocity adjoint");
        auto tGradResVelWrtConfig = mVelocityResidual.gradientConfig(aControl, aCurrentStates);
        Plato::MatrixTimesVectorPlusVector(tGradResVelWrtConfig, tCurrentVelocityAdjoint, tGradCriterionWrtConfig);

        Plato::blas1::axpy(1.0, tGradCriterionWrtConfig, aTotalDerivative);
    }

    void readTemperatureBoundaryConditions(Teuchos::ParameterList& aInputs)
    {
        Plato::ScalarVector tBcValues;
        Plato::LocalOrdinalVector tBcDofs;
        if(aInputs.isSublist("Temperature Boundary Conditions"))
        {
            auto tTempBCs = aInputs.sublist("Temperature Boundary Conditions");
            mTemperatureStateBoundaryConditions = Plato::EssentialBCs<EnergyConservationT>(tTempBCs, mSpatialModel.MeshSets);
        }
        else
        {
            THROWERR("Temperature boundary conditions are not defined for fluid mechanics problem.")
        }
    }

    void readPressureBoundaryConditions(Teuchos::ParameterList& aInputs)
    {
        Plato::ScalarVector tBcValues;
        Plato::LocalOrdinalVector tBcDofs;
        if(aInputs.isSublist("Pressure Boundary Conditions"))
        {
            auto tPressBCs = aInputs.sublist("Pressure Boundary Conditions");
            mPressureStateBoundaryConditions = Plato::EssentialBCs<MassConservationT>(tPressBCs, mSpatialModel.MeshSets);
        }
        else
        {
            THROWERR("Pressure boundary conditions are not defined for fluid mechanics problem.")
        }
    }

    void readVelocityBoundaryConditions(Teuchos::ParameterList& aInputs)
    {
        Plato::ScalarVector tBcValues;
        Plato::LocalOrdinalVector tBcDofs;
        if(aInputs.isSublist("Velocity Boundary Conditions"))
        {
            auto tVelBCs = aInputs.sublist("Velocity Boundary Conditions");
            mVelocityStateBoundaryConditions = Plato::EssentialBCs<MomentumConservationT>(tVelBCs, mSpatialModel.MeshSets);
        }
        else
        {
            THROWERR("Velocity boundary conditions are not defined for fluid mechanics problem.")
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
