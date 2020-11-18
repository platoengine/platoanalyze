/*
 * ComputationalFluidDynamicsTests.cpp
 *
 *  Created on: Oct 13, 2020
 */

#include <Teuchos_UnitTestHarness.hpp>
#include <Teuchos_XMLParameterListCoreHelpers.hpp>
#include <unordered_map>

#include <Omega_h_mark.hpp>
#include <Omega_h_shape.hpp>
#include <Omega_h_array_ops.hpp>

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

#include "PlatoTestHelpers.hpp"

namespace Plato
{

inline std::vector<std::string>
parse_criterion_names(Teuchos::ParameterList & aInputs)
{
    if(aInputs.isParameter("Functions") == false)
    {
        THROWERR(std::string("'Functions' keyword is not defined. ") +
                 "The 'Functions' keyword is used to set the names of each weighted scalar function.")
    }

    std::vector<std::string> tOutput;
    auto tNames = aInputs.get<Teuchos::Array<std::string>>("Functions").toVector();
    for (Plato::OrdinalType tIndex = 0; tIndex < tNames.size(); tIndex++)
    {
        tOutput.push_back(tNames[tIndex]);
    }
    return tOutput;
}

inline std::vector<Plato::Scalar>
parse_criterion_weights(Teuchos::ParameterList & aInputs)
{
    if(aInputs.isParameter("Weights") == false)
    {
        THROWERR(std::string("'Weights' keyword is not defined. ") +
                 "The 'Weights' keyword is used to set the weight of each weighted scalar function.")
    }

    std::vector<Plato::Scalar> tOutput;
    auto tWeights = aInputs.get<Teuchos::Array<Plato::Scalar>>("Weights").toVector();
    for (Plato::OrdinalType tIndex = 0; tIndex < tWeights.size(); tIndex++)
    {
        tOutput.push_back(tWeights[tIndex]);
    }
    return tOutput;
}

inline Omega_h::LOs
faces_on_non_prescribed_boundary
(const std::vector<std::string> & aSideSetNames,
       Omega_h::Mesh            & aMesh,
       Omega_h::MeshSets        & aMeshSets)
{
    auto tNumFaces = aMesh.nfaces();
    auto tFacesAreOnNonPrescribedBoundary = Omega_h::mark_by_class_dim(&aMesh, Omega_h::FACE, Omega_h::FACE);
    for(auto& tName : aSideSetNames)
    {
        auto tFacesOnPrescribedBoundary = Plato::side_set_face_ordinals(aMeshSets, tName);
        auto tFacesAreOnPrescribedBoundary = Omega_h::mark_image(tFacesOnPrescribedBoundary, tNumFaces);
        auto tFacesAreNotOnPrescribedBoundary = Omega_h::invert_marks(tFacesAreOnPrescribedBoundary);
        tFacesAreOnNonPrescribedBoundary = Omega_h::land_each(tFacesAreOnNonPrescribedBoundary, tFacesAreNotOnPrescribedBoundary);
    }
    // this last array has one entry for every non-traction boundary mesh face, and that entry is the face number
    auto tFacesOnNonPrescribedBoundary = Omega_h::collect_marked(tFacesAreOnNonPrescribedBoundary);
    return tFacesOnNonPrescribedBoundary;
}

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

inline std::vector<std::string>
sideset_names(Teuchos::ParameterList & aInputs)
{
    std::vector<std::string> tOutput;
    for (Teuchos::ParameterList::ConstIterator tItr = aInputs.begin(); tItr != aInputs.end(); ++tItr)
    {
        const Teuchos::ParameterEntry &tEntry = aInputs.entry(tItr);
        if (!tEntry.isList())
        {
            THROWERR("Get Side Set Names: Parameter list block is not valid.  Expect lists only.")
        }

        const std::string &tName = aInputs.name(tItr);
        if(aInputs.isSublist(tName) == false)
        {
            THROWERR(std::string("Parameter sublist with name '") + tName.c_str() + "' is not defined.")
        }

        Teuchos::ParameterList &tSubList = aInputs.sublist(tName);
        if(tSubList.isParameter("Sides") == false)
        {
            THROWERR(std::string("Keyword 'Sides' is not define in Parameter Sublist with name '") + tName.c_str() + "'.")
        }
        const auto tValue = tSubList.get<std::string>("Sides");
        tOutput.push_back(tValue);
    }
    return tOutput;
}

inline std::vector<std::string> parse_input_parameter_list
(const std::string & aTag,
 const Teuchos::ParameterList & aInputs)
{
    if(!aInputs.isParameter(aTag))
    {
        THROWERR(std::string("Parameter with tag '") + aTag + "' in block '" + aInputs.name() + "' is not defined.")
    }
    auto tSideSets = aInputs.get< Teuchos::Array<std::string> >(aTag);

    auto tLength = tSideSets.size();
    std::vector<std::string> tOutput(tLength);
    for(auto & tName : tOutput)
    {
        auto tIndex = &tName - &tOutput[0];
        tOutput[tIndex] = tSideSets[tIndex];
    }
    return tOutput;
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
        THROWERR(std::string("Dimensionless parameter with '") + aTag + "' must be defined for analysis. Use '"
            + aTag + "' keyword inside sublist 'Dimensionless Properties' to define its value")
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

struct Solutions
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
// struct Solutions

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
template <typename SimplexFadTypesT, typename T>
struct is_fad {
  static constexpr bool value = std::is_same< T, typename SimplexFadTypesT::MassFad     >::value ||
                                std::is_same< T, typename SimplexFadTypesT::ControlFad  >::value ||
                                std::is_same< T, typename SimplexFadTypesT::ConfigFad   >::value ||
                                std::is_same< T, typename SimplexFadTypesT::EnergyFad   >::value ||
                                std::is_same< T, typename SimplexFadTypesT::MomentumFad >::value;
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
    std::string mType = "undefined";
    std::unordered_map<std::string, Plato::Scalar> mScalars;
    std::unordered_map<std::string, Plato::ScalarVector> mVectors;

public:
    decltype(mType) function() const
    {
        return mType;
    }

    void function(const decltype(mType)& aInput)
    {
        mType = aInput;
    }

    Plato::Scalar scalar(const std::string& aTag) const
    {
        auto tLowerTag = Plato::tolower(aTag);
        auto tItr = mScalars.find(tLowerTag);
        if(tItr == mScalars.end())
        {
            THROWERR(std::string("State scalar with tag '") + aTag + "' is not defined in state map.")
        }
        return tItr->second;
    }

    void scalar(const std::string& aTag, const Plato::Scalar& aInput)
    {
        auto tLowerTag = Plato::tolower(aTag);
        mScalars[tLowerTag] = aInput;
    }

    Plato::ScalarVector vector(const std::string& aTag) const
    {
        auto tLowerTag = Plato::tolower(aTag);
        auto tItr = mVectors.find(tLowerTag);
        if(tItr == mVectors.end())
        {
            THROWERR(std::string("State with tag '") + aTag + "' is not defined in state map.")
        }
        return tItr->second;
    }

    void vector(const std::string& aTag, const Plato::ScalarVector& aInput)
    {
        auto tLowerTag = Plato::tolower(aTag);
        mVectors[tLowerTag] = aInput;
    }

    bool isVectorMapEmpty() const
    {
        return mVectors.empty();
    }

    bool isScalarMapEmpty() const
    {
        return mScalars.empty();
    }

    bool empty(const std::string & aTag, std::string aMap = "vector") const
    {
        auto tLowerMap = Plato::tolower(aMap);
        if(tLowerMap == "vector")
        {
            return this->isVectorDefined(aTag);
        }
        else if(tLowerMap == "scalar")
        {
            return this->isScalarDefined(aTag);
        }
        else
        {
            THROWERR(std::string("Map '") + aMap + "' is not supported.")
        }
    }

private:
    bool isScalarDefined(const std::string & aTag) const
    {
        auto tLowerTag = Plato::tolower(aTag);
        auto tItr = mScalars.find(tLowerTag);
        if(tItr == mScalars.end())
        {
            return true;
        }
        return false;
    }

    bool isVectorDefined(const std::string & aTag) const
    {
        auto tLowerTag = Plato::tolower(aTag);
        auto tItr = mVectors.find(tLowerTag);
        if(tItr == mVectors.end())
        {
            return true;
        }
        return false;
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
};

template <typename SimplexPhysicsT>
struct GradCurrentMomentumTypes : EvaluationTypes<SimplexPhysicsT>
{
  using FadType = typename Plato::FluidMechanics::SimplexFadTypes<SimplexPhysicsT>::MomentumFad;

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
  using FadType = typename Plato::FluidMechanics::SimplexFadTypes<SimplexPhysicsT>::EnergyFad;

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
  using FadType = typename Plato::FluidMechanics::SimplexFadTypes<SimplexPhysicsT>::MassFad;

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
  using FadType = typename Plato::FluidMechanics::SimplexFadTypes<SimplexPhysicsT>::MomentumFad;

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
  using FadType = typename Plato::FluidMechanics::SimplexFadTypes<SimplexPhysicsT>::EnergyFad;

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
  using FadType = typename Plato::FluidMechanics::SimplexFadTypes<SimplexPhysicsT>::MassFad;

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
  using FadType = typename Plato::FluidMechanics::SimplexFadTypes<SimplexPhysicsT>::MomentumFad;

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
   using Residual         = ResultTypes<SimplexPhysicsT>;
   using GradConfig       = GradConfigTypes<SimplexPhysicsT>;
   using GradControl      = GradControlTypes<SimplexPhysicsT>;
   using GradCurMass      = GradCurrentMassTypes<SimplexPhysicsT>;
   using GradPrevMass     = GradCurrentMassTypes<SimplexPhysicsT>;
   using GradCurEnergy    = GradCurrentEnergyTypes<SimplexPhysicsT>;
   using GradPrevEnergy   = GradPreviousEnergyTypes<SimplexPhysicsT>;
   using GradCurMomentum  = GradPreviousMomentumTypes<SimplexPhysicsT>;
   using GradPrevMomentum = GradPreviousMomentumTypes<SimplexPhysicsT>;
   using GradPredictor    = GradMomentumPredictorTypes<SimplexPhysicsT>;
};


template<typename PhysicsT, typename EvaluationT>
struct WorkSets
{
private:
    static constexpr auto mNumControls         = PhysicsT::SimplexT::mNumControls;            /*!< number of design variable fields */
    static constexpr auto mNumSpatialDims      = PhysicsT::SimplexT::mNumSpatialDims;         /*!< number of spatial dimensions */
    static constexpr auto mNumNodesPerCell     = PhysicsT::SimplexT::mNumNodesPerCell;        /*!< number of nodes per cell */
    static constexpr auto mNumPressDofsPerCell = PhysicsT::SimplexT::mNumMassDofsPerCell;     /*!< number of mass dofs per cell */
    static constexpr auto mNumTempDofsPerCell  = PhysicsT::SimplexT::mNumEnergyDofsPerCell;   /*!< number of energy dofs per cell */
    static constexpr auto mNumVelDofsPerCell   = PhysicsT::SimplexT::mNumMomentumDofsPerCell; /*!< number of momentum dofs per cell */

    Plato::OrdinalType mNumCells;

    Plato::ScalarMultiVector mTimeStep;
    Plato::ScalarMultiVector mArtificialCompressibility;

    Plato::ScalarArray3DT<typename EvaluationT::ConfigScalarType> mConfiguration;

    Plato::ScalarMultiVectorT<typename EvaluationT::ControlScalarType> mControls;
    Plato::ScalarMultiVectorT<typename EvaluationT::CurrentMassScalarType> mCurrentPress;
    Plato::ScalarMultiVectorT<typename EvaluationT::CurrentEnergyScalarType> mCurrentTemp;
    Plato::ScalarMultiVectorT<typename EvaluationT::CurrentMomentumScalarType> mCurrentVel;
    Plato::ScalarMultiVectorT<typename EvaluationT::PreviousMassScalarType> mPreviousPress;
    Plato::ScalarMultiVectorT<typename EvaluationT::PreviousEnergyScalarType> mPreviousTemp;
    Plato::ScalarMultiVectorT<typename EvaluationT::PreviousMomentumScalarType> mPreviousVel;
    Plato::ScalarMultiVectorT<typename EvaluationT::MomentumPredictorScalarType> mVelPredictor;

public:
    explicit WorkSets(const Plato::OrdinalType& aNumCells) :
        mNumCells(aNumCells),
        mTimeStep("Time Step Workset", aNumCells, mNumNodesPerCell),
        mArtificialCompressibility("Artificial Compressibility Workset", aNumCells, mNumNodesPerCell),
        mConfiguration("Configuration Workset", aNumCells, mNumNodesPerCell, mNumSpatialDims),
        mControls("Control Workset", aNumCells, mNumNodesPerCell),
        mCurrentPress("Current Mass Workset", aNumCells, mNumPressDofsPerCell),
        mCurrentTemp("Current Energy Workset", aNumCells, mNumTempDofsPerCell),
        mCurrentVel("Current Momentum Workset", aNumCells, mNumVelDofsPerCell),
        mPreviousPress("Previous Mass Workset", aNumCells, mNumPressDofsPerCell),
        mPreviousTemp("Previous Energy Workset", aNumCells, mNumTempDofsPerCell),
        mPreviousVel("Previous Momentum Workset", aNumCells, mNumVelDofsPerCell),
        mVelPredictor("Momentum Predictor Workset", aNumCells, mNumVelDofsPerCell)
    {}

    decltype(mNumCells) numCells() const
    {
        return mNumCells;
    }

    decltype(mTimeStep) timeStep()
    {
        return mTimeStep;
    }

    decltype(mArtificialCompressibility) artificialCompress()
    {
        return mArtificialCompressibility;
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

    decltype(mPreviousTemp) previousTemperature()
    {
        return mPreviousTemp;
    }

    decltype(mPreviousVel) previousVelocity()
    {
        return mPreviousVel;
    }

    decltype(mVelPredictor) predictor()
    {
        return mVelPredictor;
    }
};


// todo: abstract scalar function
template<typename PhysicsT, typename EvaluationT>
class AbstractScalarFunction
{
public:
    AbstractScalarFunction(){}
    virtual ~AbstractScalarFunction(){}

    virtual void evaluate
    (const Plato::FluidMechanics::WorkSets<PhysicsT, EvaluationT> & aWorkSets,
     Plato::ScalarVectorT<typename EvaluationT::ResultScalarType> & aResult) const = 0;

    virtual void evaluateBoundary
    (const Plato::FluidMechanics::WorkSets<PhysicsT, EvaluationT> & aWorkSets,
     Plato::ScalarVectorT<typename EvaluationT::ResultScalarType> & aResult) const = 0;
};
// class AbstractScalarFunction

template<typename PhysicsT, typename EvaluationT>
class AverageSurfacePressure : Plato::FluidMechanics::AbstractScalarFunction<PhysicsT, EvaluationT>
{
private:
    static constexpr auto mNumSpatialDims       = PhysicsT::SimplexT::mNumSpatialDims;         /*!< number of spatial dimensions */
    static constexpr auto mNumSpatialDimsOnFace = PhysicsT::SimplexT::mNumSpatialDimsOnFace;   /*!< number of spatial dimensions on face */
    static constexpr auto mNumNodesPerFace      = PhysicsT::SimplexT::mNumNodesPerFace;        /*!< number of nodes per face */
    static constexpr auto mNumPressDofsPerNode  = PhysicsT::SimplexT::mNumMassDofsPerNode;     /*!< number of energy dofs per node */

    using ResultT   = typename EvaluationT::ResultScalarType;
    using ConfigT   = typename EvaluationT::ConfigScalarType;
    using ControlT  = typename EvaluationT::ControlScalarType;
    using PressureT = typename EvaluationT::CurrentMassScalarType;

    // set local typenames
    using CubatureRule  = Plato::LinearTetCubRuleDegreeOne<mNumSpatialDimsOnFace>;
    using StateWorkSets = Plato::FluidMechanics::WorkSets<PhysicsT, EvaluationT>;

    // member metadata
    Plato::DataMap& mDataMap;
    CubatureRule mCubatureRule;
    const Plato::SpatialDomain& mSpatialDomain;

    // member parameters
    std::vector<std::string> mWallSets;

public:
    AverageSurfacePressure
    (const std::string          & aName,
     const Plato::SpatialDomain & aDomain,
     Plato::DataMap             & aDataMap,
     Teuchos::ParameterList     & aInputs) :
         mDataMap(aDataMap),
         mCubatureRule(CubatureRule()),
         mSpatialDomain(aDomain)
    {
        auto tMyCriteria = aInputs.sublist("Criteria").sublist(aName);
        mWallSets = Plato::parse_input_parameter_list("Wall", tMyCriteria);
    }

    virtual ~AverageSurfacePressure(){}

    void evaluate(const StateWorkSets & aWorkSets, Plato::ScalarVectorT<ResultT> & aResult) const
    { return; }

    void evaluateBoundary(const StateWorkSets & aWorkSets, Plato::ScalarVectorT<ResultT> & aResult) const
    {
        // set face to element graph
        auto tFace2eElems      = mSpatialDomain.Mesh.ask_up(mNumSpatialDimsOnFace, mNumSpatialDims);
        auto tFace2Elems_map   = tFace2eElems.a2ab;
        auto tFace2Elems_elems = tFace2eElems.ab2b;

        // set mesh vertices
        auto tFace2Verts = mSpatialDomain.Mesh.ask_verts_of(mNumSpatialDimsOnFace);
        auto tCell2Verts = mSpatialDomain.Mesh.ask_elem_verts();

        // allocate local functors
        Plato::NodeCoordinate<mNumSpatialDims> tCoords(&(mSpatialDomain.Mesh));
        Plato::ComputeSurfaceJacobians<mNumSpatialDims> tComputeSurfaceJacobians;
        Plato::ComputeSurfaceIntegralWeight<mNumSpatialDims> tComputeSurfaceIntegralWeight;
        Plato::InterpolateFromNodal<mNumSpatialDims, mNumPressDofsPerNode> tIntrplScalarField;
        Plato::CreateFaceLocalNode2ElemLocalNodeIndexMap<mNumSpatialDims> tCreateFaceLocalNode2ElemLocalNodeIndexMap;

        // set local data
        auto tCubatureWeight = mCubatureRule.getCubWeight();
        auto tBasisFunctions = mCubatureRule.getBasisFunctions();

        // set local worksets
        auto tNumCells = mSpatialDomain.Mesh.nelems();
        Plato::ScalarVectorT<PressureT> tCurrentPressGP("current pressure at Gauss point", tNumCells);

        // set input worksets
        auto tCurrentPressureWS = aWorkSets.currentPressure();
        auto tConfigurationWS   = aWorkSets.configuration();

        for(auto& tName : mWallSets)
        {
            // get faces on this side set
            auto tFaceOrdinalsOnSideSet = Plato::side_set_face_ordinals(mSpatialDomain.MeshSets, tName);
            auto tNumFaces = tFaceOrdinalsOnSideSet.size();
            Plato::ScalarArray3DT<ConfigT> tJacobians("face Jacobians", tNumFaces, mNumSpatialDimsOnFace, mNumSpatialDims);

            Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumFaces), LAMBDA_EXPRESSION(const Plato::OrdinalType & aFaceI)
            {
                auto tFaceOrdinal = tFaceOrdinalsOnSideSet[aFaceI];
                // for all elements connected to this face, which is either 1 or 2 elements
                for( Plato::OrdinalType tElem = tFace2Elems_map[tFaceOrdinal]; tElem < tFace2Elems_map[tFaceOrdinal+1]; tElem++ )
                {
                    // create a map from face local node index to elem local node index
                    Plato::OrdinalType tLocalNodeOrd[mNumSpatialDims];
                    auto tCellOrdinal = tFace2Elems_elems[tElem];
                    tCreateFaceLocalNode2ElemLocalNodeIndexMap(tCellOrdinal, tFaceOrdinal, tCell2Verts, tFace2Verts, tLocalNodeOrd);

                    // calculate surface Jacobian and surface integral weight
                    ConfigT tSurfaceAreaTimesCubWeight(0.0);
                    tComputeSurfaceJacobians(tCellOrdinal, aFaceI, tLocalNodeOrd, tConfigurationWS, tJacobians);
                    tComputeSurfaceIntegralWeight(aFaceI, tCubatureWeight, tJacobians, tSurfaceAreaTimesCubWeight);

                    // evaluate surface scalar function
                    tIntrplScalarField(tCellOrdinal, tBasisFunctions, tCurrentPressureWS, tCurrentPressGP);

                    // calculate surface integral, which is defined as
                    // \int_{\Gamma_e}N_p^a p^h d\Gamma_e
                    for( Plato::OrdinalType tNode=0; tNode < mNumNodesPerFace; tNode++)
                    {
                        for( Plato::OrdinalType tDof=0; tDof < mNumPressDofsPerNode; tDof++)
                        {
                            auto tDofOrdinal = tLocalNodeOrd[tNode] * mNumPressDofsPerNode + tDof;
                            aResult(tCellOrdinal, tDofOrdinal) += tBasisFunctions(tNode) *
                                tCurrentPressGP(tCellOrdinal) * tSurfaceAreaTimesCubWeight;
                        }
                    }
                }
            }, "average surface pressure integral");

        }
    }
};
// class AverageSurfacePressure




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
    using PrimalStates = Plato::FluidMechanics::States;

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


// todo: physics scalar function
template<typename PhysicsT>
class PhysicsScalarFunction : public Plato::FluidMechanics::CriterionBase
{
private:
    std::string mFuncName;

    static constexpr auto mNumControlsPerNode     = PhysicsT::SimplexT::mNumControl;             /*!< number of design variable fields */
    static constexpr auto mNumSpatialDims         = PhysicsT::SimplexT::mNumSpatialDims;         /*!< number of spatial dimensions */
    static constexpr auto mNumNodesPerCell        = PhysicsT::SimplexT::mNumNodesPerCell;        /*!< number of nodes per cell */
    static constexpr auto mNumMassDofsPerCell     = PhysicsT::SimplexT::mNumMassDofsPerCell;     /*!< number of mass dofs per cell */
    static constexpr auto mNumEnergyDofsPerCell   = PhysicsT::SimplexT::mNumEnergyDofsPerCell;   /*!< number of energy dofs per cell */
    static constexpr auto mNumMomentumDofsPerCell = PhysicsT::SimplexT::mNumMomentumDofsPerCell; /*!< number of momentum dofs per cell */
    static constexpr auto mNumMassDofsPerNode     = PhysicsT::SimplexT::mNumMassDofsPerNode;     /*!< number of mass dofs per node */
    static constexpr auto mNumEnergyDofsPerNode   = PhysicsT::SimplexT::mNumEnergyDofsPerNode;   /*!< number of energy dofs per node */
    static constexpr auto mNumMomentumDofsPerNode = PhysicsT::SimplexT::mNumMomentumDofsPerNode; /*!< number of momentum dofs per node */
    static constexpr auto mNumConfigDofsPerCell   = mNumSpatialDims * mNumNodesPerCell;          /*!< number of configuration degrees of freedom per cell */

    // forward automatic differentiation evaluation types
    using ResidualEvalT     = typename Plato::FluidMechanics::Evaluation<typename PhysicsT::SimplexT>::Residual;
    using GradConfigEvalT   = typename Plato::FluidMechanics::Evaluation<typename PhysicsT::SimplexT>::GradConfig;
    using GradControlEvalT  = typename Plato::FluidMechanics::Evaluation<typename PhysicsT::SimplexT>::GradControl;
    using GradCurVelEvalT   = typename Plato::FluidMechanics::Evaluation<typename PhysicsT::SimplexT>::GradCurMomentum;
    using GradCurTempEvalT  = typename Plato::FluidMechanics::Evaluation<typename PhysicsT::SimplexT>::GradCurEnergy;
    using GradCurPressEvalT = typename Plato::FluidMechanics::Evaluation<typename PhysicsT::SimplexT>::GradCurMass;

    // element scalar functions types
    using ResidualFunc     = std::shared_ptr<Plato::FluidMechanics::AbstractScalarFunction<typename PhysicsT::SimplexT, ResidualEvalT>>;
    using GradConfigFunc   = std::shared_ptr<Plato::FluidMechanics::AbstractScalarFunction<typename PhysicsT::SimplexT, GradConfigEvalT>>;
    using GradControlFunc  = std::shared_ptr<Plato::FluidMechanics::AbstractScalarFunction<typename PhysicsT::SimplexT, GradControlEvalT>>;
    using GradCurVelFunc   = std::shared_ptr<Plato::FluidMechanics::AbstractScalarFunction<typename PhysicsT::SimplexT, GradCurVelEvalT>>;
    using GradCurTempFunc  = std::shared_ptr<Plato::FluidMechanics::AbstractScalarFunction<typename PhysicsT::SimplexT, GradCurTempEvalT>>;
    using GradCurPressFunc = std::shared_ptr<Plato::FluidMechanics::AbstractScalarFunction<typename PhysicsT::SimplexT, GradCurPressEvalT>>;

    // element scalar functions per element block, i.e. domain
    std::unordered_map<std::string, ResidualFunc>     mResidualFuncs;
    std::unordered_map<std::string, GradConfigFunc>   mGradConfigFuncs;
    std::unordered_map<std::string, GradControlFunc>  mGradControlFuncs;
    std::unordered_map<std::string, GradCurVelFunc>   mGradCurrMomentumFuncs;
    std::unordered_map<std::string, GradCurTempFunc>  mGradCurrEnergyFuncs;
    std::unordered_map<std::string, GradCurPressFunc> mGradCurrMassFuncs;

    Plato::DataMap& mDataMap;
    const Plato::SpatialModel& mSpatialModel;
    Plato::NodeCoordinate<mNumSpatialDims> mNodeCoordinate;  /*!< node coordinates metadata */

    // local-to-global physics degrees of freedom maps (note: both pressure and temperature state have 1 dofs per node.)
    Plato::VectorEntryOrdinal<mNumSpatialDims, mNumSpatialDims>         mConfigEntryOrdinal;      /*!< configuration local-to-global ID map */
    Plato::VectorEntryOrdinal<mNumSpatialDims, mNumControlsPerNode>      mScalarEntryOrdinal;      /*!< control local-to-global ID map */
    Plato::VectorEntryOrdinal<mNumSpatialDims, mNumMomentumDofsPerCell> mVectorStateEntryOrdinal; /*!< momentum state local-to-global ID map */
    Plato::VectorEntryOrdinal<mNumSpatialDims, mNumMassDofsPerCell>     mScalarStateEntryOrdinal; /*!< mass state local-to-global ID map */

    // define local type names
    using PrimalStates        = Plato::FluidMechanics::States;
    using ResidualWorkSets    = Plato::FluidMechanics::WorkSets<PhysicsT, ResidualEvalT>;
    using GradVelWorkSets     = Plato::FluidMechanics::WorkSets<PhysicsT, GradCurVelEvalT>;
    using GradTempWorkSets    = Plato::FluidMechanics::WorkSets<PhysicsT, GradCurTempEvalT>;
    using GradPressWorkSets   = Plato::FluidMechanics::WorkSets<PhysicsT, GradCurPressEvalT>;
    using GradConfigWorkSets  = Plato::FluidMechanics::WorkSets<PhysicsT, GradConfigEvalT>;
    using GradControlWorkSets = Plato::FluidMechanics::WorkSets<PhysicsT, GradControlEvalT>;

public:
    PhysicsScalarFunction
    (const Plato::SpatialModel & aSpatialModel,
     Plato::DataMap & aDataMap,
     Teuchos::ParameterList & aInputs,
     std::string & aName):
        mSpatialModel(aSpatialModel),
        mDataMap     (aDataMap),
        mFuncName    (aName)
    {
        this->initialize(aInputs);
    }

    virtual ~PhysicsScalarFunction(){}

    std::string name() const
    {
        return mFuncName;
    }

    Plato::Scalar value
    (const Plato::ScalarVector & aControls,
     const PrimalStates & aStates) const
    {
        ResidualEvalT tReturnValue(0.0);
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            ResidualWorkSets tWorkSets(tNumCells);
            this->setValueWorkSets(tDomain, aControls, aStates, tWorkSets);

            auto tName = tDomain.getDomainName();
            using ResultScalarT = typename ResidualEvalT::ResultScalarType;
            Plato::ScalarVectorT<ResultScalarT> tResultWS("cells value", tNumCells);
            mResidualFuncs.at(tName)->evaluate(tWorkSets, tResultWS);

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
            GradConfigWorkSets tWorkSets(tNumCells);
            this->setGradConfigWorkSets(tDomain, aControls, aStates, tWorkSets);

            auto tName = tDomain.getDomainName();
            using ResultScalarT = typename GradConfigEvalT::ResultScalarType;
            Plato::ScalarVectorT<ResultScalarT> tResultWS("cells value", tNumCells);
            mResidualFuncs.at(tName)->evaluate(tWorkSets, tResultWS);

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
        Plato::ScalarVector tGradient("gradient wrt control", mNumControlsPerNode * tNumNodes);
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            GradControlWorkSets tWorkSets(tNumCells);
            this->setGradControlWorkSets(tDomain, aControls, aStates, tWorkSets);

            auto tName = tDomain.getDomainName();
            using ResultScalarT = typename GradControlEvalT::ResultScalarType;
            Plato::ScalarVectorT<ResultScalarT> tResultWS("cells value", tNumCells);
            mResidualFuncs.at(tName)->evaluate(tWorkSets, tResultWS);

            Plato::assemble_vector_gradient_fad<mNumNodesPerCell, mNumControlsPerNode>
                (tDomain, mScalarEntryOrdinal, tResultWS, tGradient);
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
            GradPressWorkSets tWorkSets(tNumCells);
            this->setGradPressWorkSets(tDomain, aControls, aStates, tWorkSets);

            auto tName = tDomain.getDomainName();
            using ResultScalarT = typename GradCurPressEvalT::ResultScalarType;
            Plato::ScalarVectorT<ResultScalarT> tResultWS("cells value", tNumCells);
            mResidualFuncs.at(tName)->evaluate(tWorkSets, tResultWS);

            Plato::assemble_vector_gradient_fad<mNumNodesPerCell, mNumMassDofsPerNode>
                (tDomain, mScalarStateEntryOrdinal, tResultWS, tGradient);
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
            GradTempWorkSets tWorkSets(tNumCells);
            this->setGradTempWorkSets(tDomain, aControls, aStates, tWorkSets);

            auto tName = tDomain.getDomainName();
            using ResultScalarT = typename GradCurTempEvalT::ResultScalarType;
            Plato::ScalarVectorT<ResultScalarT> tResultWS("cells value", tNumCells);
            mResidualFuncs.at(tName)->evaluate(tWorkSets, tResultWS);

            Plato::assemble_vector_gradient_fad<mNumNodesPerCell, mNumEnergyDofsPerNode>
                (tDomain, mScalarStateEntryOrdinal, tResultWS, tGradient);
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
            GradVelWorkSets tWorkSets(tNumCells);
            this->setGradVelWorkSets(tDomain, aControls, aStates, tWorkSets);

            auto tName = tDomain.getDomainName();
            using ResultScalarT = typename GradCurVelEvalT::ResultScalarType;
            Plato::ScalarVectorT<ResultScalarT> tResultWS("cells value", tNumCells);
            mResidualFuncs.at(tName)->evaluate(tWorkSets, tResultWS);

            Plato::assemble_vector_gradient_fad<mNumNodesPerCell, mNumMomentumDofsPerNode>
                (tDomain, mVectorStateEntryOrdinal, tResultWS, tGradient);
        }

        return tGradient;
    }

private:
    void initialize(Teuchos::ParameterList & aInputs)
    {
        typename PhysicsT::FunctionFactory tScalarFuncFactory;

        auto tInputs = aInputs.sublist("Criteria").sublist(mFuncName);
        auto tFuncType = tInputs.get<std::string>("Scalar Function Type", "not defined");

        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tName = tDomain.getDomainName();

            mResidualFuncs[tName] =
                tScalarFuncFactory.template createScalarFunction<ResidualEvalT>
                    (tFuncType, mFuncName, tDomain, mDataMap, aInputs);

            mGradConfigFuncs[tName] =
                tScalarFuncFactory.template createScalarFunction<GradConfigEvalT>
                    (tFuncType, mFuncName, tDomain, mDataMap, aInputs);

            mGradControlFuncs[tName] =
                tScalarFuncFactory.template createScalarFunction<GradControlEvalT>
                    (tFuncType, mFuncName, tDomain, mDataMap, aInputs);

            mGradCurrMassFuncs[tName] =
                tScalarFuncFactory.template createScalarFunction<GradCurPressEvalT>
                    (tFuncType, mFuncName, tDomain, mDataMap, aInputs);

            mGradCurrEnergyFuncs[tName] =
                tScalarFuncFactory.template createScalarFunction<GradCurTempEvalT>
                    (tFuncType, mFuncName, tDomain, mDataMap, aInputs);

            mGradCurrMomentumFuncs[tName] =
                tScalarFuncFactory.template createScalarFunction<GradCurVelEvalT>
                    (tFuncType, mFuncName, tDomain, mDataMap, aInputs);
        }
    }

    void setValueWorkSets
    (const Plato::SpatialDomain & aDomain,
     const Plato::ScalarVector  & aControls,
     const PrimalStates         & aState,
           ResidualWorkSets     & aWorkSets) const
    {
        Plato::workset_state_scalar_scalar<mNumMomentumDofsPerNode, mNumNodesPerCell>
            (aDomain, mVectorStateEntryOrdinal, aState.vector("current velocity"), aWorkSets.currentVelocity());

        Plato::workset_state_scalar_scalar<mNumMassDofsPerNode, mNumNodesPerCell>
            (aDomain, mScalarStateEntryOrdinal, aState.vector("current pressure"), aWorkSets.currentPressure());

        Plato::workset_state_scalar_scalar<mNumEnergyDofsPerNode, mNumNodesPerCell>
            (aDomain, mScalarStateEntryOrdinal, aState.vector("current temperature"), aWorkSets.currentTemperature());

        Plato::workset_control_scalar_scalar<mNumNodesPerCell>
            (aDomain, mScalarEntryOrdinal, aControls, aWorkSets.control());

        Plato::workset_control_scalar_scalar<mNumNodesPerCell>
            (aDomain, mScalarEntryOrdinal, aState.vector("time step"), aWorkSets.timeStep());

        Plato::workset_config_scalar<mNumSpatialDims, mNumNodesPerCell>
            (aDomain, mNodeCoordinate, aWorkSets.configuration());
    }

    void setGradConfigWorkSets
    (const Plato::SpatialDomain & aDomain,
     const Plato::ScalarVector  & aControls,
     const PrimalStates         & aState,
           GradConfigWorkSets   & aWorkSets) const
    {
        Plato::workset_state_scalar_scalar<mNumMomentumDofsPerNode, mNumNodesPerCell>
            (aDomain, mVectorStateEntryOrdinal, aState.vector("current velocity"), aWorkSets.currentVelocity());

        Plato::workset_state_scalar_scalar<mNumMassDofsPerNode, mNumNodesPerCell>
            (aDomain, mScalarStateEntryOrdinal, aState.vector("current pressure"), aWorkSets.currentPressure());

        Plato::workset_state_scalar_scalar<mNumEnergyDofsPerNode, mNumNodesPerCell>
            (aDomain, mScalarStateEntryOrdinal, aState.vector("current temperature"), aWorkSets.currentTemperature());

        Plato::workset_control_scalar_scalar<mNumNodesPerCell>
            (aDomain, mScalarEntryOrdinal, aControls, aWorkSets.control());

        Plato::workset_control_scalar_scalar<mNumNodesPerCell>
            (aDomain, mScalarEntryOrdinal, aState.vector("time step"), aWorkSets.timeStep());

        using ConfigScalarT = typename GradConfigEvalT::ConfigScalarType;
        Plato::workset_config_fad<mNumSpatialDims, mNumNodesPerCell, mNumConfigDofsPerCell, ConfigScalarT>
            (aDomain, mNodeCoordinate, aWorkSets.configuration());
    }

    void setGradControlWorkSets
    (const Plato::SpatialDomain & aDomain,
     const Plato::ScalarVector  & aControls,
     const PrimalStates         & aState,
           GradControlWorkSets  & aWorkSets) const
    {
        Plato::workset_state_scalar_scalar<mNumMomentumDofsPerNode, mNumNodesPerCell>
            (aDomain, mVectorStateEntryOrdinal, aState.vector("current velocity"), aWorkSets.currentVelocity());

        Plato::workset_state_scalar_scalar<mNumMassDofsPerNode, mNumNodesPerCell>
            (aDomain, mScalarStateEntryOrdinal, aState.vector("current pressure"), aWorkSets.currentPressure());

        Plato::workset_state_scalar_scalar<mNumEnergyDofsPerNode, mNumNodesPerCell>
            (aDomain, mScalarStateEntryOrdinal, aState.vector("current temperature"), aWorkSets.currentTemperature());

        using ControlScalarT = typename GradControlEvalT::ControlScalarType;
        Plato::workset_control_scalar_fad<mNumNodesPerCell, ControlScalarT>
            (aDomain, mScalarEntryOrdinal, aControls, aWorkSets.control());

        Plato::workset_control_scalar_scalar<mNumNodesPerCell>
            (aDomain, mScalarEntryOrdinal, aState.vector("time step"), aWorkSets.timeStep());

        Plato::workset_config_scalar<mNumSpatialDims, mNumNodesPerCell, mNumConfigDofsPerCell>
            (aDomain, mNodeCoordinate, aWorkSets.configuration());
    }

    void setGradPressWorkSets
    (const Plato::SpatialDomain & aDomain,
     const Plato::ScalarVector  & aControls,
     const PrimalStates         & aState,
           GradPressWorkSets    & aWorkSets) const
    {
        Plato::workset_state_scalar_scalar<mNumMomentumDofsPerNode, mNumNodesPerCell>
            (aDomain, mVectorStateEntryOrdinal, aState.vector("current velocity"), aWorkSets.currentVelocity());

        using PressScalarT = typename GradCurPressEvalT::CurrentMassScalarType;
        Plato::workset_state_scalar_fad<mNumMassDofsPerNode, mNumNodesPerCell, PressScalarT>
            (aDomain, mScalarStateEntryOrdinal, aState.vector("current pressure"), aWorkSets.currentPressure());

        Plato::workset_state_scalar_scalar<mNumEnergyDofsPerNode, mNumNodesPerCell>
            (aDomain, mScalarStateEntryOrdinal, aState.vector("current temperature"), aWorkSets.currentTemperature());

        Plato::workset_control_scalar_scalar<mNumNodesPerCell>
            (aDomain, mScalarEntryOrdinal, aControls, aWorkSets.control());

        Plato::workset_control_scalar_scalar<mNumNodesPerCell>
            (aDomain, mScalarEntryOrdinal, aState.vector("time step"), aWorkSets.timeStep());

        Plato::workset_config_scalar<mNumSpatialDims, mNumNodesPerCell, mNumConfigDofsPerCell>
            (aDomain, mNodeCoordinate, aWorkSets.configuration());
    }

    void setGradTempWorkSets
    (const Plato::SpatialDomain & aDomain,
     const Plato::ScalarVector  & aControls,
     const PrimalStates         & aState,
           GradTempWorkSets     & aWorkSets) const
    {
        Plato::workset_state_scalar_scalar<mNumMomentumDofsPerNode, mNumNodesPerCell>
            (aDomain, mVectorStateEntryOrdinal, aState.vector("current velocity"), aWorkSets.currentVelocity());

        Plato::workset_state_scalar_scalar<mNumMassDofsPerNode, mNumNodesPerCell>
            (aDomain, mScalarStateEntryOrdinal, aState.vector("current pressure"), aWorkSets.currentPressure());

        using TempScalarT = typename GradCurTempEvalT::CurrentEnergyScalarType;
        Plato::workset_state_scalar_fad<mNumEnergyDofsPerNode, mNumNodesPerCell, TempScalarT>
            (aDomain, mScalarStateEntryOrdinal, aState.vector("current temperature"), aWorkSets.currentTemperature());

        Plato::workset_control_scalar_scalar<mNumNodesPerCell>
            (aDomain, mScalarEntryOrdinal, aControls, aWorkSets.control());

        Plato::workset_control_scalar_scalar<mNumNodesPerCell>
            (aDomain, mScalarEntryOrdinal, aState.vector("time step"), aWorkSets.timeStep());

        Plato::workset_config_scalar<mNumSpatialDims, mNumNodesPerCell, mNumConfigDofsPerCell>
            (aDomain, mNodeCoordinate, aWorkSets.configuration());
    }

    void setGradVelWorkSets
    (const Plato::SpatialDomain & aDomain,
     const Plato::ScalarVector  & aControls,
     const PrimalStates         & aState,
           GradVelWorkSets      & aWorkSets) const
    {
        using VelScalarT = typename GradCurVelEvalT::CurrentMomentumScalarType;
        Plato::workset_state_scalar_fad<mNumMomentumDofsPerNode, mNumNodesPerCell, VelScalarT>
            (aDomain, mVectorStateEntryOrdinal, aState.vector("current velocity"), aWorkSets.currentVelocity());

        Plato::workset_state_scalar_scalar<mNumMassDofsPerNode, mNumNodesPerCell>
            (aDomain, mScalarStateEntryOrdinal, aState.vector("current pressure"), aWorkSets.currentPressure());

        Plato::workset_state_scalar_scalar<mNumEnergyDofsPerNode, mNumNodesPerCell>
            (aDomain, mScalarStateEntryOrdinal, aState.vector("current temperature"), aWorkSets.currentTemperature());

        Plato::workset_control_scalar_scalar<mNumNodesPerCell>
            (aDomain, mScalarEntryOrdinal, aControls, aWorkSets.control());

        Plato::workset_control_scalar_scalar<mNumNodesPerCell>
            (aDomain, mScalarEntryOrdinal, aState.vector("time step"), aWorkSets.timeStep());

        Plato::workset_config_scalar<mNumSpatialDims, mNumNodesPerCell, mNumConfigDofsPerCell>
            (aDomain, mNodeCoordinate, aWorkSets.configuration());
    }
};
// class PhysicsScalarFunction










template<typename PhysicsT, typename EvaluationT>
class PressureSurfaceForces
{
private:
    static constexpr auto mNumDofsPerNode = PhysicsT::mNumDofsPerNode; /*!< number of degrees of freedom (dofs) per node */

    static constexpr auto mNumSpatialDims       = PhysicsT::SimplexT::mNumSpatialDims;       /*!< number of spatial dimensions */
    static constexpr auto mNumNodesPerFace      = PhysicsT::SimplexT::mNumNodesPerFace;      /*!< number of nodes per face */
    static constexpr auto mNumPressDofsPerNode  = PhysicsT::SimplexT::mNumMassDofsPerNode;   /*!< number of pressure dofs per node */
    static constexpr auto mNumSpatialDimsOnFace = PhysicsT::SimplexT::mNumSpatialDimsOnFace; /*!< number of spatial dimensions on face */

    // forward automatic differentiation types
    using ResultT    = typename EvaluationT::ResultScalarType;
    using ConfigT    = typename EvaluationT::ConfigScalarType;
    using PrevPressT = typename EvaluationT::PreviousMassScalarType;

    const std::string mSideSetName; /*!< side set name */

    const Plato::SpatialDomain& mSpatialDomain; /*!< Plato spatial model */
    Plato::LinearTetCubRuleDegreeOne<mNumSpatialDimsOnFace> mCubatureRule;  /*!< integration rule */

public:
    PressureSurfaceForces
    (const Plato::SpatialDomain & aSpatialDomain,
     std::string aSideSetName = "empty") :
         mSideSetName(aSideSetName),
         mSpatialDomain(aSpatialDomain)
    {
    }

    void operator()
    (const Plato::FluidMechanics::WorkSets<PhysicsT, EvaluationT> & aWorkSets,
     Plato::ScalarMultiVectorT<ResultT> & aResult,
     Plato::Scalar aMultiplier = 1.0) const
    {
        // get mesh vertices
        auto tFace2Verts = mSpatialDomain.Mesh.ask_verts_of(mNumSpatialDimsOnFace);
        auto tCell2Verts = mSpatialDomain.Mesh.ask_elem_verts();

        // get face to element graph
        auto tFace2eElems = mSpatialDomain.Mesh.ask_up(mNumSpatialDimsOnFace, mNumSpatialDims);
        auto tFace2Elems_map   = tFace2eElems.a2ab;
        auto tFace2Elems_elems = tFace2eElems.ab2b;

        // get element to face map
        auto tElem2Faces = mSpatialDomain.Mesh.ask_down(mNumSpatialDims, mNumSpatialDimsOnFace).ab2b;

        // set local functors
        Plato::NodeCoordinate<mNumSpatialDims> tCoords(&(mSpatialDomain.Mesh));
        Plato::ComputeSurfaceJacobians<mNumSpatialDims> tComputeSurfaceJacobians;
        Plato::ComputeSurfaceIntegralWeight<mNumSpatialDims> tComputeSurfaceIntegralWeight;
        Plato::InterpolateFromNodal<mNumSpatialDims, mNumPressDofsPerNode> tIntrplScalarField;
        Plato::CreateFaceLocalNode2ElemLocalNodeIndexMap<mNumSpatialDims> tCreateFaceLocalNode2ElemLocalNodeIndexMap;

        // get sideset faces
        auto tFaceLocalOrdinals = Plato::side_set_face_ordinals(mSpatialDomain.MeshSets, mSideSetName);
        auto tNumFaces = tFaceLocalOrdinals.size();
        Plato::ScalarArray3DT<ConfigT> tSurfaceJacobians("jacobian", tNumFaces, mNumSpatialDimsOnFace, mNumSpatialDims);

        // set previous pressure at Gauss points container
        auto tNumCells = mSpatialDomain.Mesh.nelems();
        Plato::ScalarVectorT<PrevPressT> tPrevPressGP("previous pressure at Gauss point", tNumCells);

        // set input state worksets
        auto tConfigWS    = aWorkSets.configuration();
        auto tPrevPressWS = aWorkSets.previousPressure();

        // calculate surface integral
        auto tCubatureWeight = mCubatureRule.getCubWeight();
        auto tBasisFunctions = mCubatureRule.getBasisFunctions();
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumFaces), LAMBDA_EXPRESSION(const Plato::OrdinalType & aFaceI)
        {

          auto tFaceOrdinal = tFaceLocalOrdinals[aFaceI];
          // for each element that the face is connected to: (either 1 or 2 elements)
          for( Plato::OrdinalType tElem = tFace2Elems_map[tFaceOrdinal]; tElem < tFace2Elems_map[tFaceOrdinal + 1]; tElem++ )
          {
              // create a map from face local node index to elem local node index
              Plato::OrdinalType tLocalNodeOrd[mNumSpatialDims];
              auto tCellOrdinal = tFace2Elems_elems[tElem];
              tCreateFaceLocalNode2ElemLocalNodeIndexMap(tCellOrdinal, tFaceOrdinal, tCell2Verts, tFace2Verts, tLocalNodeOrd);

              // calculate surface jacobians
              ConfigT tSurfaceAreaTimesCubWeight(0.0);
              tComputeSurfaceJacobians(tCellOrdinal, aFaceI, tLocalNodeOrd, tConfigWS, tSurfaceJacobians);
              tComputeSurfaceIntegralWeight(aFaceI, tCubatureWeight, tSurfaceJacobians, tSurfaceAreaTimesCubWeight);

              // compute unit normal vector
              auto tElemFaceOrdinal = Plato::get_face_ordinal<mNumSpatialDims>(tCellOrdinal, tFaceOrdinal, tElem2Faces);
              auto tUnitNormalVec = Plato::unit_normal_vector(tCellOrdinal, tElemFaceOrdinal, tCoords);

              // project into aResult workset
              tIntrplScalarField(tCellOrdinal, tBasisFunctions, tPrevPressWS, tPrevPressGP);
              for( Plato::OrdinalType tNode = 0; tNode < mNumNodesPerFace; tNode++ )
              {
                  for( Plato::OrdinalType tDof = 0; tDof < mNumDofsPerNode; tDof++ )
                  {
                      auto tCellDofOrdinal = (tLocalNodeOrd[tNode] * mNumDofsPerNode) + tDof;
                      aResult(tCellOrdinal, tCellDofOrdinal) += aMultiplier * tBasisFunctions(tNode) *
                          tUnitNormalVec(tDof) * tPrevPressGP(tCellOrdinal) * tSurfaceAreaTimesCubWeight;
                  }
              }
          }
        }, "calculate surface pressure integral");
    }
};
// class PressureSurfaceForces




template<Plato::OrdinalType NumNodesPerCell,
         Plato::OrdinalType NumSpaceDim,
         typename StateT,
         typename ConfigT,
         typename ResultT>
DEVICE_TYPE void calculate_strain_rate
(const Plato::OrdinalType & aCellOrdinal,
 const StateT  & aStateWS,
 const ConfigT & aGradient,
       ResultT & aStrainRate)
{
    // calculate strain rate for incompressible flows, which is defined as
    // \frac{1}{2}\left( \frac{\partial u_i}{\partial x_j} + \frac{\partial u_j}{\partial x_i} \right)
    for(Plato::OrdinalType tNode = 0; tNode < NumNodesPerCell; tNode++)
    {
        for(Plato::OrdinalType tDimI = 0; tDimI < NumSpaceDim; tDimI++)
        {
            for(Plato::OrdinalType tDimJ = 0; tDimJ < NumSpaceDim; tDimJ++)
            {
                aStrainRate(aCellOrdinal, tDimI, tDimJ) += static_cast<Plato::Scalar>(0.5) *
                    ( ( aGradient(aCellOrdinal, tNode, tDimJ) * aStateWS(aCellOrdinal, tDimI) )
                    + ( aGradient(aCellOrdinal, tNode, tDimI) * aStateWS(aCellOrdinal, tDimJ) ) );
            }
        }
    }
}

template<typename PhysicsT, typename EvaluationT>
class DeviatoricSurfaceForces
{
private:
    static constexpr auto mNumDofsPerNode = PhysicsT::mNumDofsPerNode; /*!< number of degrees of freedom per node */
    static constexpr auto mNumDofsPerCell = PhysicsT::mNumDofsPerCell; /*!< number of degrees of freedom per cell */

    static constexpr auto mNumSpatialDims       = PhysicsT::SimplexT::mNumSpatialDims;       /*!< number of spatial dimensions */
    static constexpr auto mNumNodesPerFace      = PhysicsT::SimplexT::mNumNodesPerFace;      /*!< number of nodes per face */
    static constexpr auto mNumNodesPerCell      = PhysicsT::SimplexT::mNumNodesPerCell;      /*!< number of nodes per cell */
    static constexpr auto mNumSpatialDimsOnFace = PhysicsT::SimplexT::mNumSpatialDimsOnFace; /*!< number of spatial dimensions on face */

    // forward automatic differentiation types
    using ResultT  = typename EvaluationT::ResultScalarType;
    using ConfigT  = typename EvaluationT::ConfigScalarType;
    using ControlT = typename EvaluationT::ControlScalarType;
    using PrevVelT = typename EvaluationT::PreviousMomentumScalarType;

    using StrainT = typename Plato::FluidMechanics::fad_type_t<typename PhysicsT::SimplexT, PrevVelT, ConfigT>;

    Plato::Scalar mPrNum = 1.0;
    Plato::Scalar mPrNumConvexityParam = 0.5;
    std::string mSideSetName = ""; /*!< side set name */

    Omega_h::LOs mBoundaryFaceOrdinals;
    const Plato::SpatialDomain& mSpatialDomain; /*!< Plato spatial model */
    Plato::LinearTetCubRuleDegreeOne<mNumSpatialDims> mCubatureRule;  /*!< integration rule */

public:
    DeviatoricSurfaceForces
    (const Plato::SpatialDomain & aSpatialDomain,
     Teuchos::ParameterList & aInputs,
     std::string aSideSetName = "") :
         mSideSetName(aSideSetName),
         mSpatialDomain(aSpatialDomain)
    {
        this->setPenaltyModel(aInputs);
        this->setDimensionlessProperties(aInputs);
        this->setFacesOnNonPrescribedBoundary(aInputs);
    }

    void operator()
    (const Plato::FluidMechanics::WorkSets<PhysicsT, EvaluationT> & aWorkSets,
           Plato::ScalarMultiVectorT<ResultT>                     & aResult,
           Plato::Scalar                                            aMultiplier = 1.0) const
    {
        // get mesh vertices
        auto tFace2Verts = mSpatialDomain.Mesh.ask_verts_of(mNumSpatialDimsOnFace);
        auto tCell2Verts = mSpatialDomain.Mesh.ask_elem_verts();

        // get face to element graph
        auto tFace2eElems = mSpatialDomain.Mesh.ask_up(mNumSpatialDimsOnFace, mNumSpatialDims);
        auto tFace2Elems_map   = tFace2eElems.a2ab;
        auto tFace2Elems_elems = tFace2eElems.ab2b;

        // get element to face map
        auto tElem2Faces = mSpatialDomain.Mesh.ask_down(mNumSpatialDims, mNumSpatialDimsOnFace).ab2b;

        // set local functors
        Plato::ComputeGradientWorkset<mNumSpatialDims> tComputeGradient;
        Plato::NodeCoordinate<mNumSpatialDims> tCoords(&(mSpatialDomain.Mesh));
        Plato::ComputeSurfaceJacobians<mNumSpatialDims> tComputeSurfaceJacobians;
        Plato::InterpolateFromNodal<mNumSpatialDims, mNumDofsPerNode> tIntrplVectorField;
        Plato::ComputeSurfaceIntegralWeight<mNumSpatialDims> tComputeSurfaceIntegralWeight;
        Plato::CreateFaceLocalNode2ElemLocalNodeIndexMap<mNumSpatialDims> tCreateFaceLocalNode2ElemLocalNodeIndexMap;

        // transfer member data to device
        auto tPrNum = mPrNum;
        auto tPrNumConvexityParam = mPrNumConvexityParam;
        auto tBoundaryFaceOrdinals = mBoundaryFaceOrdinals;

        // set local data structures
        auto tNumCells = mSpatialDomain.Mesh.nelems();
        Plato::ScalarVectorT<ConfigT>  tCellVolume("cell weight", tNumCells);
        Plato::ScalarArray3DT<ConfigT> tGradient("cell gradient", tNumCells, mNumNodesPerCell, mNumSpatialDims);
        Plato::ScalarArray3DT<StrainT> tStrainRate("cell strain rate", tNumCells, mNumSpatialDims, mNumSpatialDims);
        auto tNumFaces = tBoundaryFaceOrdinals.size();
        Plato::ScalarArray3DT<ConfigT> tJacobians("cell jacobians", tNumFaces, mNumSpatialDimsOnFace, mNumSpatialDims);

        // set input state worksets
        auto tControlWS    = aWorkSets.control();
        auto tConfigWS     = aWorkSets.configuration();
        auto tPrevVelWS    = aWorkSets.previousVelocity();

        // calculate surface integral
        auto tCubatureWeight = mCubatureRule.getCubWeight();
        auto tBasisFunctions = mCubatureRule.getBasisFunctions();
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumFaces), LAMBDA_EXPRESSION(const Plato::OrdinalType & aFaceI)
        {
            auto tFaceOrdinal = tBoundaryFaceOrdinals[aFaceI];
            // for each element that the face is connected to: (either 1 or 2 elements)
            for( Plato::OrdinalType tElem = tFace2Elems_map[tFaceOrdinal]; tElem < tFace2Elems_map[tFaceOrdinal + 1]; tElem++ )
            {
                // create a map from face local node index to elem local node index
                Plato::OrdinalType tLocalNodeOrd[mNumSpatialDims];
                auto tCellOrdinal = tFace2Elems_elems[tElem];
                tCreateFaceLocalNode2ElemLocalNodeIndexMap(tCellOrdinal, tFaceOrdinal, tCell2Verts, tFace2Verts, tLocalNodeOrd);

                // calculate surface jacobians
                ConfigT tSurfaceAreaTimesCubWeight(0.0);
                tComputeSurfaceJacobians(tCellOrdinal, aFaceI, tLocalNodeOrd, tConfigWS, tJacobians);
                tComputeSurfaceIntegralWeight(aFaceI, tCubatureWeight, tJacobians, tSurfaceAreaTimesCubWeight);

                // compute unit normal vector
                auto tElemFaceOrdinal = Plato::get_face_ordinal<mNumSpatialDims>(tCellOrdinal, tFaceOrdinal, tElem2Faces);
                auto tUnitNormalVec = Plato::unit_normal_vector(tCellOrdinal, tElemFaceOrdinal, tCoords);

                // calculate strain rate
                tComputeGradient(tCellOrdinal, tGradient, tConfigWS, tCellVolume);
                Plato::FluidMechanics::calculate_strain_rate<mNumNodesPerCell, mNumSpatialDims>
                    (tCellOrdinal, tPrevVelWS, tGradient, tStrainRate);

                // calculate penalized prandtl number
                ControlT tDensity = Plato::cell_density<mNumNodesPerCell>(tCellOrdinal, tControlWS);
                ControlT tPenalizedPrandtlNum = ( tDensity * ( tPrNum * (1.0 - tPrNumConvexityParam) - 1.0 ) + 1.0 )
                    / ( tPrNum * (1.0 + tPrNumConvexityParam * tDensity) );

                // calculate deviatoric traction forces, which are defined as,
                // \int_{\Gamma_e} N_u^a \left( \tau^h_{ij}n_j \right) d\Gamma_e
                for(Plato::OrdinalType tNode = 0; tNode < mNumNodesPerFace; tNode++)
                {
                    for(Plato::OrdinalType tDimI = 0; tDimI < mNumSpatialDims; tDimI++)
                    {
                        auto tDof = (mNumSpatialDims * tNode) + tDimI;
                        for(Plato::OrdinalType tDimJ = 0; tDimJ < mNumSpatialDims; tDimJ++)
                        {
                            aResult(tCellOrdinal, tDof) += tBasisFunctions(tNode) * tSurfaceAreaTimesCubWeight *
                                aMultiplier * ( ( static_cast<Plato::Scalar>(2.0) * tPenalizedPrandtlNum *
                                    tStrainRate(tCellOrdinal, tDimI, tDimJ) ) * tUnitNormalVec(tDimJ) );
                        }
                    }
                }
            }

        }, "calculate deviatoric traction integral");
    }

private:
    void setPenaltyModel
    (Teuchos::ParameterList & aInputs)
    {
        if(aInputs.isSublist("Hyperbolic"))
        {
            auto tHyperbolicList = aInputs.sublist("Hyperbolic");
            if(tHyperbolicList.isSublist("Penalty Function"))
            {
                auto tPenaltyFuncList = tHyperbolicList.sublist("Penalty Function");
                mPrNumConvexityParam = tPenaltyFuncList.get<Plato::Scalar>("Prandtl Number Convexity Parameter", 0.5);
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
            mPrNum = Plato::parse_dimensionless_property<Plato::Scalar>(tSublist, "Prandtl Number");
        }
        else
        {
            THROWERR("'Dimensionless Properties' block is not defined.")
        }
    }

    void setFacesOnNonPrescribedBoundary(Teuchos::ParameterList& aInputs)
    {
        if(mSideSetName.empty())
        {
            if(aInputs.isSublist("Momentum Natural Boundary Conditions"))
            {
                auto tNaturalBCs = aInputs.sublist("Momentum Natural Boundary Conditions");
                auto tNames = Plato::sideset_names(tNaturalBCs);
                mBoundaryFaceOrdinals =
                    Plato::faces_on_non_prescribed_boundary(tNames, mSpatialDomain.Mesh, mSpatialDomain.MeshSets);
            }
            else
            {
                THROWERR(std::string("Expected to deduce stabilized momentum residual boundary surfaces from the ")
                    + "'Momentum Natural Boundary Conditions' block defined in the input file.  However, this block "
                    + "is not defined in the input file.")
            }
        }
        else
        {
            mBoundaryFaceOrdinals = Plato::side_set_face_ordinals(mSpatialDomain.MeshSets, mSideSetName);
        }
    }
};
// class DeviatoricSurfaceForces

// todo: abstract vector function
template<typename PhysicsT, typename EvaluationT>
class AbstractVectorFunction
{
public:
    AbstractVectorFunction(){}
    virtual ~AbstractVectorFunction(){}

    virtual void evaluate
    (const Plato::FluidMechanics::WorkSets<PhysicsT, EvaluationT>      & aWorkSets,
     Plato::ScalarMultiVectorT<typename EvaluationT::ResultScalarType> & aResult) const = 0;

    virtual void evaluateBoundary
    (const Plato::FluidMechanics::WorkSets<PhysicsT, EvaluationT>      & aWorkSets,
     Plato::ScalarMultiVectorT<typename EvaluationT::ResultScalarType> & aResult) const = 0;

    virtual void evaluatePrescribed
    (const Plato::FluidMechanics::WorkSets<PhysicsT, EvaluationT>      & aWorkSets,
     Plato::ScalarMultiVectorT<typename EvaluationT::ResultScalarType> & aResult) const = 0;
};
// class AbstractVectorFunction

template<typename PhysicsT, typename EvaluationT>
class VelocityPredictorResidual :
    public Plato::FluidMechanics::AbstractVectorFunction<PhysicsT, EvaluationT>
{
private:
    static constexpr auto mNumDofsPerNode = PhysicsT::mNumDofsPerNode; /*!< number of degrees of freedom per node */
    static constexpr auto mNumDofsPerCell = PhysicsT::mNumDofsPerCell; /*!< number of degrees of freedom per cell */

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

    Plato::DataMap& mDataMap; /*!< output database */
    const Plato::SpatialDomain& mSpatialDomain; /*!< Plato spatial model */
    Plato::LinearTetCubRuleDegreeOne<mNumSpatialDims> mCubatureRule; /*!< cubature rule evaluator */

    // set local type names
    using StateWorkSets    = Plato::FluidMechanics::WorkSets<PhysicsT, EvaluationT>;
    using PressureForces   = Plato::FluidMechanics::PressureSurfaceForces<PhysicsT, EvaluationT>;
    using DeviatoricForces = Plato::FluidMechanics::DeviatoricSurfaceForces<PhysicsT, EvaluationT>;

    // set external force evaluators
    std::unordered_map<std::string, std::shared_ptr<PressureForces>>     mPressureBCs;   /*!< prescribed pressure forces */
    std::unordered_map<std::string, std::shared_ptr<DeviatoricForces>>   mDeviatoricBCs; /*!< stabilized deviatoric boundary forces */
    std::shared_ptr<Plato::NaturalBCs<mNumSpatialDims, mNumDofsPerNode>> mPrescribedBCs; /*!< prescribed boundary conditions, e.g. tractions */

    // set member scalar data
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
        this->setPenaltyModel(aInputs);
        this->setDimensionlessProperties(aInputs);
        this->setNaturalBoundaryConditions(aInputs);
        this->checkNaturalBoundaryConditions(aInputs);
    }

    virtual ~VelocityPredictorResidual(){}

    void evaluate
    (const StateWorkSets                & aWorkSets,
     Plato::ScalarMultiVectorT<ResultT> & aResult) const
    {
        using StrainT =
            typename Plato::FluidMechanics::fad_type_t<typename PhysicsT::SimplexT, PrevVelT, ConfigT>;

        auto tNumCells = aResult.extent(0);
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
        Plato::InterpolateFromNodal<mNumSpatialDims, mNumTempDofsPerNode> tIntrplScalarField;
        Plato::InterpolateFromNodal<mNumSpatialDims, mNumVelDofsPerNode, 0/*offset*/, mNumSpatialDims> tIntrplVectorField;

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

            // calculate convective force integral, which are defined as
            // \int_{\Omega_e} N_u^a \left( \frac{\partial}{\partial x_j}(u^{n-1}_j u^{n-1}_i) \right) d\Omega_e
            tIntrplVectorField(aCellOrdinal, tBasisFunctions, tPrevVelWS, tPrevVelGP);
            for(Plato::OrdinalType tNode = 0; tNode < mNumNodesPerCell; tNode++)
            {
                for(Plato::OrdinalType tDimI = 0; tDimI < mNumSpatialDims; tDimI++)
                {
                    auto tDofIndex = (mNumSpatialDims * tNode) + tDimI;
                    for(Plato::OrdinalType tDimJ = 0; tDimJ < mNumSpatialDims; tDimJ++)
                    {
                        aResult(aCellOrdinal, tDofIndex) += tCellVolume(aCellOrdinal) * tBasisFunctions(tNode) *
                            ( tGradient(aCellOrdinal, tNode, tDimJ) *  ( tPrevVelGP(aCellOrdinal, tDimJ) *
                                tPrevVelGP(aCellOrdinal, tDimI) ) );

                        tStabForce(aCellOrdinal, tDimI) += tGradient(aCellOrdinal, tNode, tDimJ) *
                            ( tPrevVelGP(aCellOrdinal, tDimJ) * tPrevVelGP(aCellOrdinal, tDimI) );
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
            ControlT tPenalizedPrandtlNum = ( tDensity * ( tPrNum * (1.0 - tPrNumConvexityParam) - 1.0 ) + 1.0 )
                / ( tPrNum * (1.0 + tPrNumConvexityParam * tDensity) );

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
            tIntrplScalarField(aCellOrdinal, tBasisFunctions, tPrevTempWS, tPrevTempGP);
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
            // \int_{\Omega_e} N_u^a (\frac{Pr}{Da} u^{n-1}_i) d\Omega
            ControlT tPenalizedBrinkmanCoeff = (tPrNum / tDaNum) * (1.0 - tDensity) / (1.0 + (tBrinkmanConvexityParam * tDensity));
            for(Plato::OrdinalType tNode = 0; tNode < mNumNodesPerCell; tNode++)
            {
                for(Plato::OrdinalType tDim = 0; tDim < mNumSpatialDims; tDim++)
                {
                    auto tDofIndex = (mNumSpatialDims * tNode) + tDim;
                    aResult(aCellOrdinal, tDofIndex) += tCellVolume(aCellOrdinal) *
                        tBasisFunctions(tNode) * (tPenalizedBrinkmanCoeff * tPrevVelGP(aCellOrdinal, tDim));
                    tStabForce(aCellOrdinal, tDim) += tPenalizedBrinkmanCoeff * tPrevVelGP(aCellOrdinal, tDim);
                }
            }

            // calculate stabilizing force integral, which are defined as
            // \int_{\Omega_e} \left( \frac{\partial N_u^a}{\partial x_k} u^{n-1}_k \right) F_i^{stab} d\Omega_e
            // where the stabilizing force, F_i^{stab}, is defined as
            // F_i^{stab} = \frac{\partial}{\partial x_j}(u^{n-1}_j u^{n-1}_i) + Gr_i Pr^2 T^h + \frac{Pr}{Da} u^{n-1}_i
            for(Plato::OrdinalType tNode = 0; tNode < mNumNodesPerCell; tNode++)
            {
                for(Plato::OrdinalType tDimI = 0; tDimI < mNumSpatialDims; tDimI++)
                {
                    auto tDofIndex = (mNumSpatialDims * tNode) + tDimI;
                    for(Plato::OrdinalType tDimJ = 0; tDimJ < mNumSpatialDims; tDimJ++)
                    {
                        aResult(aCellOrdinal, tDofIndex) += static_cast<Plato::Scalar>(0.5) * tTimeStepWS(aCellOrdinal, tNode) *
                            tCellVolume(aCellOrdinal) * tBasisFunctions(tNode) * ( tGradient(aCellOrdinal, tNode, tDimJ) *
                                tPrevVelGP(aCellOrdinal, tDimJ) ) * tStabForce(aCellOrdinal, tDimI);
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
            // \int_{Omega_e} N_u^a \left( u^\ast_i - u^{n-1}_i \right) d\Omega_e
            tIntrplVectorField(aCellOrdinal, tBasisFunctions, tPredictorWS, tPredictorGP);
            for(Plato::OrdinalType tNode = 0; tNode < mNumNodesPerCell; tNode++)
            {
                for(Plato::OrdinalType tDimI = 0; tDimI < mNumSpatialDims; tDimI++)
                {
                    auto tDofIndex = (mNumSpatialDims * tNode) + tDimI;
                    aResult(aCellOrdinal, tDofIndex) += tCellVolume(aCellOrdinal) * tBasisFunctions(tNode) *
                        ( tPredictorGP(aCellOrdinal, tDimI) - tPrevVelGP(aCellOrdinal, tDimI) );
                }
            }

        }, "velocity predictor residual");
    }

   void evaluateBoundary
   (const StateWorkSets                & aWorkSets,
    Plato::ScalarMultiVectorT<ResultT> & aResult) const
   {
       // calculate boundary integral, which is defined as
       // \int_{\Gamma-\Gamma_t} N_u^a\left(\tau_{ij}n_j\right) d\Gamma
       auto tNumCells = aResult.extent(0);
       Plato::ScalarMultiVectorT<ResultT> tResultWS("deviatoric forces", tNumCells, mNumDofsPerCell);
       for(auto& tPair : mDeviatoricBCs)
       {
           tPair.second->operator()(aWorkSets, tResultWS);
       }

       // multiply force vector by the corresponding nodal time steps
       auto tTimeStepWS = aWorkSets.timeStep();
       Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
       {
           for(Plato::OrdinalType tNode = 0; tNode < mNumNodesPerCell; tNode++)
           {
               for(Plato::OrdinalType tDof = 0; tDof < mNumDofsPerNode; tDof++)
               {
                   auto tDofIndex = (mNumDofsPerNode * tNode) + tDof;
                   aResult(aCellOrdinal, tDofIndex) += tTimeStepWS(aCellOrdinal, tNode) *
                       static_cast<Plato::Scalar>(-1.0) * tResultWS(aCellOrdinal, tDofIndex);
               }
           }
       }, "deviatoric traction forces");
   }

   void evaluatePrescribed
   (const StateWorkSets                & aWorkSets,
    Plato::ScalarMultiVectorT<ResultT> & aResult) const
   {
       if( mPrescribedBCs != nullptr )
       {
           // set input worksets
           auto tConfigWS   = aWorkSets.configuration();
           auto tControlWS  = aWorkSets.control();
           auto tPrevVelWS  = aWorkSets.previousVelocity();

           // calculate deviatoric traction forces, which are defined as
           // \int_{\Gamma_t} N_u^a\left(t_i + p^{n-1}n_i\right) d\Gamma
           auto tNumCells = aResult.extent(0);
           Plato::ScalarMultiVectorT<ResultT> tResultWS("traction forces", tNumCells, mNumDofsPerCell);
           mPrescribedBCs->get( mSpatialDomain, tPrevVelWS, tControlWS, tConfigWS, tResultWS);
           for(auto& tPair : mPressureBCs)
           {
               tPair.second->operator()(aWorkSets, tResultWS);
           }

           // multiply force vector by the corresponding nodal time steps
           auto tTimeStepWS = aWorkSets.timeStep();
           Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
           {
               for(Plato::OrdinalType tNode = 0; tNode < mNumNodesPerCell; tNode++)
               {
                   for(Plato::OrdinalType tDof = 0; tDof < mNumDofsPerNode; tDof++)
                   {
                       auto tDofIndex = (mNumDofsPerNode * tNode) + tDof;
                       aResult(aCellOrdinal, tDofIndex) += tTimeStepWS(aCellOrdinal, tNode) *
                           static_cast<Plato::Scalar>(-1.0) * tResultWS(aCellOrdinal, tDofIndex);
                   }
               }
           }, "prescribed traction forces");
       }
   }

private:
   void setPenaltyModel
   (Teuchos::ParameterList & aInputs)
   {
       if(aInputs.isSublist("Hyperbolic"))
       {
           auto tHyperbolicList = aInputs.sublist("Hyperbolic");
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
            THROWERR("'Dimensionless Properties' block is not defined.")
        }
    }

    void setNaturalBoundaryConditions(Teuchos::ParameterList& aInputs)
    {
        if(aInputs.isSublist("Momentum Natural Boundary Conditions"))
        {
            auto tInputsNaturalBCs = aInputs.sublist("Momentum Natural Boundary Conditions");
            mPrescribedBCs = std::make_shared<Plato::NaturalBCs<mNumSpatialDims, mNumDofsPerNode>>(tInputsNaturalBCs);

            for (Teuchos::ParameterList::ConstIterator tItr = tInputsNaturalBCs.begin(); tItr != tInputsNaturalBCs.end(); ++tItr)
            {
                const Teuchos::ParameterEntry &tEntry = tInputsNaturalBCs.entry(tItr);
                if (!tEntry.isList())
                {
                    THROWERR("Get Side Set Names: Parameter list block is not valid.  Expect lists only.")
                }

                const std::string &tName = tInputsNaturalBCs.name(tItr);
                if(tInputsNaturalBCs.isSublist(tName) == false)
                {
                    THROWERR(std::string("Parameter sublist with name '") + tName.c_str() + "' is not defined.")
                }

                Teuchos::ParameterList &tSubList = tInputsNaturalBCs.sublist(tName);
                if(tSubList.isParameter("Sides") == false)
                {
                    THROWERR(std::string("Keyword 'Sides' is not define in Parameter Sublist with name '") + tName.c_str() + "'.")
                }
                const auto tSideSetName = tSubList.get<std::string>("Sides");

                auto tNaturalBC = std::make_shared<PressureForces>(mSpatialDomain, tSideSetName);
                mPressureBCs.insert(std::make_pair<std::string, std::shared_ptr<PressureForces>>(tSideSetName, tNaturalBC));
            }
        }
    }

    void setStabilizedNaturalBoundaryConditions(Teuchos::ParameterList& aInputs)
    {
        if(aInputs.isSublist("Balancing Momentum Natural Boundary Conditions"))
        {
            auto tInputsNaturalBCs = aInputs.sublist("Balancing Momentum Natural Boundary Conditions");
            for (Teuchos::ParameterList::ConstIterator tItr = tInputsNaturalBCs.begin(); tItr != tInputsNaturalBCs.end(); ++tItr)
            {
                const Teuchos::ParameterEntry &tEntry = tInputsNaturalBCs.entry(tItr);
                if (!tEntry.isList())
                {
                    THROWERR("Get Side Set Names: Parameter list block is not valid.  Expect lists only.")
                }

                const std::string &tName = tInputsNaturalBCs.name(tItr);
                if(tInputsNaturalBCs.isSublist(tName) == false)
                {
                    THROWERR(std::string("Parameter sublist with name '") + tName.c_str() + "' is not defined.")
                }

                Teuchos::ParameterList &tSubList = tInputsNaturalBCs.sublist(tName);
                if(tSubList.isParameter("Sides") == false)
                {
                    THROWERR(std::string("Keyword 'Sides' is not define in Parameter Sublist with name '") + tName.c_str() + "'.")
                }
                const auto tSideSetName = tSubList.get<std::string>("Sides");

                auto tNaturalBC = std::make_shared<DeviatoricForces>(mSpatialDomain, tSideSetName);
                mDeviatoricBCs.insert(std::make_pair<std::string, std::shared_ptr<DeviatoricForces>>(tSideSetName, tNaturalBC));
            }
        }
        else
        {
            std::string tSideSetName = "Will be Deduced From Prescribed Natural Boundary Conditions";
            auto tNaturalBC = std::make_shared<DeviatoricForces>(mSpatialDomain, tSideSetName);
            mDeviatoricBCs.insert(std::make_pair<std::string, std::shared_ptr<DeviatoricForces>>(tSideSetName, tNaturalBC));
        }
    }


    void checkNaturalBoundaryConditions(Teuchos::ParameterList& aInputs)
    {
        auto tPrescribedNaturalBCsDefined = aInputs.isSublist("Momentum Natural Boundary Conditions");
        auto tStabilizedNaturalBCsDefined = aInputs.isSublist("Balancing Momentum Natural Boundary Conditions");
        if(!tPrescribedNaturalBCsDefined && !tStabilizedNaturalBCsDefined)
        {
            THROWERR(std::string("Balancing momentum forces side sets should be defined inside the 'Balancing Momentum ")
                + "Natural Boundary Conditions' block if prescribed momentum natural boundary conditions side sets are "
                + "not defined, i.e. the 'Momentum Natural Boundary Conditions' block is not defined.")
        }
    }
};
// class VelocityPredictorResidual

template<typename PhysicsT, typename EvaluationT>
class VelocityIncrementResidual :
    public Plato::FluidMechanics::AbstractVectorFunction<PhysicsT, EvaluationT>
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
    using CurVelT    = typename EvaluationT::CurrentMomentumScalarType;
    using CurPressT  = typename EvaluationT::CurrentMassScalarType;
    using PrevVelT   = typename EvaluationT::PreviousMomentumScalarType;
    using PrevPressT = typename EvaluationT::PreviousMassScalarType;
    using PredictorT = typename EvaluationT::MomentumPredictorScalarType;

    Plato::DataMap& mDataMap;                   /*!< output database */
    const Plato::SpatialDomain& mSpatialDomain; /*!< Plato spatial model */
    Plato::LinearTetCubRuleDegreeOne<mNumSpatialDims> mCubatureRule; /*!< integration rule */

    using StateWorkSets = typename Plato::FluidMechanics::WorkSets<PhysicsT, EvaluationT>;

    Plato::Scalar mThetaTwo = 0.0;

public:
    VelocityIncrementResidual
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

    virtual ~VelocityIncrementResidual(){}

    void evaluate
    (const StateWorkSets                & aWorkSets,
     Plato::ScalarMultiVectorT<ResultT> & aResult) const
    {
        auto tNumCells = aResult.extent(0);
        Plato::ScalarVectorT<ConfigT>    tCellVolume("cell weight", tNumCells);
        Plato::ScalarVectorT<CurPressT>  tCurPressGP("current pressure at Gauss point", tNumCells);
        Plato::ScalarVectorT<PrevPressT> tPrevPressGP("previous pressure at Gauss point", tNumCells);

        Plato::ScalarMultiVectorT<ResultT>    tStabForce("stabilized force", tNumCells, mNumVelDofsPerNode);
        Plato::ScalarMultiVectorT<PredictorT> tPredictorGP("corrector at Gauss point", tNumCells, mNumVelDofsPerNode);
        Plato::ScalarMultiVectorT<CurVelT>    tCurVelGP("current velocity at Gauss point", tNumCells, mNumVelDofsPerNode);
        Plato::ScalarMultiVectorT<PrevVelT>   tPrevVelGP("previous velocity at Gauss point", tNumCells, mNumVelDofsPerNode);

        Plato::ScalarArray3DT<ConfigT> tGradient("cell gradient", tNumCells, mNumNodesPerCell, mNumSpatialDims);

        // set local functors
        Plato::ComputeGradientWorkset<mNumSpatialDims> tComputeGradient;
        Plato::InterpolateFromNodal<mNumSpatialDims, mNumPressDofsPerNode> tIntrplScalarField;
        Plato::InterpolateFromNodal<mNumSpatialDims, mNumVelDofsPerNode, 0/*offset*/, mNumSpatialDims> tIntrplVectorField;

        // set input state worksets
        auto tConfigWS    = aWorkSets.configuration();
        auto tTimeStepWS  = aWorkSets.timeStep();
        auto tCurVelWS    = aWorkSets.currentVelocity();
        auto tPrevVelWS   = aWorkSets.previousVelocity();
        auto tCurPressWS  = aWorkSets.currentPressure();
        auto tPrevPressWS = aWorkSets.previousPressure();
        auto tPredictorWS = aWorkSets.predictor();

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

                    auto tDofIndex = (mNumVelDofsPerNode * tNode) + tDim;
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
                    auto tDofIndex = (mNumVelDofsPerNode * tNode) + tDimI;
                    for(Plato::OrdinalType tDimJ = 0; tDimJ < mNumSpatialDims; tDimJ++)
                    {
                        aResult(aCellOrdinal, tDofIndex) += static_cast<Plato::Scalar>(0.5) * tTimeStepWS(aCellOrdinal, tNode) *
                            tCellVolume(aCellOrdinal) * tBasisFunctions(tNode) * ( tGradient(aCellOrdinal, tNode, tDimJ) *
                                tPrevVelGP(aCellOrdinal, tDimJ) ) * tStabForce(aCellOrdinal, tDimI);
                    }
                }
            }

            // apply time step multiplier to internal force plus stabilized force vector,
            // i.e. F = \Delta{t} * \left( F_i^{int} + F_i^{stab} \right)
            for(Plato::OrdinalType tNode = 0; tNode < mNumNodesPerCell; tNode++)
            {
                for(Plato::OrdinalType tDimI = 0; tDimI < mNumSpatialDims; tDimI++)
                {
                    auto tDofIndex = (mNumVelDofsPerNode * tNode) + tDimI;
                        aResult(aCellOrdinal, tDofIndex) *= tTimeStepWS(aCellOrdinal, tNode);
                }
            }

            // calculate inertial force integral, which are defined as
            // \int_{Omega_e} N_u^a \left( u^{n}_i - u^{*}_i \right) d\Omega_e
            tIntrplVectorField(aCellOrdinal, tBasisFunctions, tCurVelWS, tCurVelGP);
            tIntrplVectorField(aCellOrdinal, tBasisFunctions, tPredictorWS, tPredictorGP);
            for(Plato::OrdinalType tNode = 0; tNode < mNumNodesPerCell; tNode++)
            {
                for(Plato::OrdinalType tDimI = 0; tDimI < mNumSpatialDims; tDimI++)
                {
                    auto tDofIndex = (mNumVelDofsPerNode * tNode) + tDimI;
                    aResult(aCellOrdinal, tDofIndex) += tCellVolume(aCellOrdinal) * tBasisFunctions(tNode) *
                        ( tCurVelGP(aCellOrdinal, tDimI) - tPredictorGP(aCellOrdinal, tDimI) );
                }
            }

        }, "velocity corrector residual");
    }

    void evaluateBoundary
    (const StateWorkSets                & aWorkSets,
     Plato::ScalarMultiVectorT<ResultT> & aResult) const
    { return; /* boundary integral equates zero */ }

    void evaluatePrescribed
    (const StateWorkSets                & aWorkSets,
     Plato::ScalarMultiVectorT<ResultT> & aResult) const
    { return; /* prescribed force integral equates zero */ }
};
// class VelocityIncrementResidual

template<typename PhysicsT, typename EvaluationT>
class TemperatureIncrementResidual :
    public Plato::FluidMechanics::AbstractVectorFunction<PhysicsT, EvaluationT>
{
private:
    static constexpr auto mNumDofsPerNode = PhysicsT::mNumDofsPerNode; /*!< number of degrees of freedom per node */
    static constexpr auto mNumDofsPerCell = PhysicsT::mNumDofsPerCell; /*!< number of degrees of freedom per cell */

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

    Plato::LinearTetCubRuleDegreeOne<mNumSpatialDims> mCubatureRule; /*!< integration rule */

    std::shared_ptr<Plato::NaturalBCs<mNumSpatialDims, mNumDofsPerNode>> mHeatFlux; /*!< heat flux evaluator */

    using StateWorkSets = Plato::FluidMechanics::WorkSets<PhysicsT, EvaluationT>;

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
         mHeatSource(0.0)
    {
        // todo: read thermal source scalar
        if(aInputs.isSublist("Thermal Natural Boundary Conditions"))
        {
            auto tSublist = aInputs.sublist("Thermal Natural Boundary Conditions");
            mHeatFlux = std::make_shared<Plato::NaturalBCs<mNumSpatialDims, mNumDofsPerNode>>(tSublist);
        }
    }

    virtual ~TemperatureIncrementResidual(){}

    void evaluate
    (const StateWorkSets                & aWorkSets,
     Plato::ScalarMultiVectorT<ResultT> & aResult) const
    {
        // set local forward ad type
        using StabForceT = typename Plato::FluidMechanics::fad_type_t<typename PhysicsT::SimplexT, PrevVelT, ConfigT, PrevTempT>;

        // set local data
        auto tNumCells = aResult.extent(0);
        Plato::ScalarVectorT<ConfigT>   tCellVolume("cell weight", tNumCells);
        Plato::ScalarVectorT<CurTempT>  tCurTempGP("current temperature at Gauss point", tNumCells);
        Plato::ScalarVectorT<PrevTempT> tPrevTempGP("previous temperature at Gauss point", tNumCells);

        Plato::ScalarMultiVectorT<StabForceT> tStabForce("stabilized force", tNumCells, mNumNodesPerCell);
        Plato::ScalarMultiVectorT<PrevVelT>   tPrevVelGP("previous velocity at Gauss point", tNumCells, mNumVelDofsPerNode);
        Plato::ScalarMultiVectorT<ResultT>    tPrevThermalGradGP("previous thermal gradient at Gauss point", tNumCells, mNumVelDofsPerNode);

        Plato::ScalarArray3DT<ConfigT> tGradient("cell gradient", tNumCells, mNumNodesPerCell, mNumSpatialDims);

        // set local functors
        Plato::ComputeGradientWorkset<mNumSpatialDims> tComputeGradient;
        Plato::InterpolateFromNodal<mNumSpatialDims, mNumTempDofsPerNode> tIntrplScalarField;
        Plato::InterpolateFromNodal<mNumSpatialDims, mNumVelDofsPerNode, 0/*offset*/, mNumSpatialDims> tIntrplVectorField;

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
            // \int_{\Omega_e} N_T^a \left( u_i^{n-1} \frac{\partial T^{n-1}}{\partial x_i} \right) d\Omega_e
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
            // \pi_T(\theta) \frac{\partial T^{n-1}}{\partial x_i} = \pi_T(\theta)\partial_i T^{n-1}
            for(Plato::OrdinalType tNode = 0; tNode < mNumNodesPerCell; tNode++)
            {
                for(Plato::OrdinalType tDim = 0; tDim < mNumSpatialDims; tDim++)
                {
                    tPrevThermalGradGP(aCellOrdinal, tDim) += tPenalizedThermalDiff *
                        tGradient(aCellOrdinal, tNode, tDim) * tPrevTempGP(aCellOrdinal);
                }
            }

            // calculate diffusive force integral, which is defined as
            // int_{\Omega_e} \frac{partial N_T^a}{\partial x_i} \left(\pi_T(\theta)\partial_i T^{n-1}\right) d\Omega_e
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
            // where F^{stab} = u_i^{n-1}\frac{\partial T^{n-1}}{\partial x_i} - \beta Q
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
            // \int_{Omega_e} N_T^a \left( T^{n} - T^{n-1} \right) d\Omega_e
            tIntrplVectorField(aCellOrdinal, tBasisFunctions, tCurTempWS, tCurTempGP);
            for(Plato::OrdinalType tNode = 0; tNode < mNumNodesPerCell; tNode++)
            {
                aResult(aCellOrdinal, tNode) += tCellVolume(aCellOrdinal) * tBasisFunctions(tNode) *
                    (tCurTempGP(aCellOrdinal) - tPrevTempGP(aCellOrdinal));
            }

        }, "conservation of energy internal forces");
    }

    void evaluateBoundary
    (const StateWorkSets                & aWorkSets,
     Plato::ScalarMultiVectorT<ResultT> & aResult) const
    { return; /* boundary integral equates zero */ }

    void evaluatePrescribed
    (const StateWorkSets                & aWorkSets,
     Plato::ScalarMultiVectorT<ResultT> & aResult) const
    {
        if( mHeatFlux != nullptr )
        {
            // set input state worksets
            auto tConfigWS   = aWorkSets.configuration();
            auto tControlWS  = aWorkSets.control();
            auto tPrevTempWS = aWorkSets.previousTemperature();

            // evaluate prescribed flux
            auto tNumCells = aResult.extent(0);
            Plato::ScalarMultiVectorT<ResultT> tResultWS("heat flux", tNumCells, mNumDofsPerCell);
            mHeatFlux.get( mSpatialDomain, tPrevTempWS, tControlWS, tConfigWS, tResultWS, -1.0 );

            auto tTimeStepWS = aWorkSets.timeStep();
            Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
            {
                for(Plato::OrdinalType tDof = 0; tDof < mNumDofsPerCell; tDof++)
                {
                    aResult(aCellOrdinal, tDof) += tTimeStepWS(aCellOrdinal, tDof) * tResultWS(aCellOrdinal, tDof);
                }
            }, "heat flux contribution");
        }
    }
};
// class TemperatureIncrementResidual



template<typename PhysicsT, typename EvaluationT>
class MomentumSurfaceForces
{
private:
    static constexpr auto mNumDofsPerNode  = PhysicsT::mNumDofsPerNode; /*!< number of degrees of freedom (dofs) per node */

    static constexpr auto mNumSpatialDims       = PhysicsT::SimplexT::mNumSpatialDims;       /*!< number of spatial dimensions */
    static constexpr auto mNumNodesPerFace      = PhysicsT::SimplexT::mNumNodesPerFace;      /*!< number of nodes per face */
    static constexpr auto mNumSpatialDimsOnFace = PhysicsT::SimplexT::mNumSpatialDimsOnFace; /*!< number of spatial dimensions on face */

    // forward automatic differentiation types
    using ResultT    = typename EvaluationT::ResultScalarType;
    using ConfigT    = typename EvaluationT::ConfigScalarType;
    using PrevVelT   = typename EvaluationT::PreviousMomentumScalarType;

    const std::string mSideSetName; /*!< side set name */

    const Plato::SpatialDomain& mSpatialDomain; /*!< Plato spatial model */
    Plato::LinearTetCubRuleDegreeOne<mNumSpatialDims> mCubatureRule;  /*!< integration rule */

public:
    MomentumSurfaceForces
    (const Plato::SpatialDomain & aSpatialDomain,
     std::string aSideSetName = "empty") :
         mSideSetName(aSideSetName),
         mSpatialDomain(aSpatialDomain)
    {
    }

    void operator()
    (const Plato::FluidMechanics::WorkSets<PhysicsT, EvaluationT> & aWorkSets,
           Plato::ScalarMultiVectorT<ResultT>                     & aResult,
           Plato::Scalar                                            aMultiplier = 1.0) const
    {
        // get mesh vertices
        auto tFace2Verts = mSpatialDomain.Mesh.ask_verts_of(mNumSpatialDimsOnFace);
        auto tCell2Verts = mSpatialDomain.Mesh.ask_elem_verts();

        // get face to element graph
        auto tFace2eElems = mSpatialDomain.Mesh.ask_up(mNumSpatialDimsOnFace, mNumSpatialDims);
        auto tFace2Elems_map   = tFace2eElems.a2ab;
        auto tFace2Elems_elems = tFace2eElems.ab2b;

        // get element to face map
        auto tElem2Faces = mSpatialDomain.Mesh.ask_down(mNumSpatialDims, mNumSpatialDimsOnFace).ab2b;

        // set local functors
        Plato::NodeCoordinate<mNumSpatialDims> tCoords(&(mSpatialDomain.Mesh));
        Plato::ComputeSurfaceJacobians<mNumSpatialDims> tComputeSurfaceJacobians;
        Plato::InterpolateFromNodal<mNumSpatialDims, mNumDofsPerNode> tIntrplVectorField;
        Plato::ComputeSurfaceIntegralWeight<mNumSpatialDims> tComputeSurfaceIntegralWeight;
        Plato::CreateFaceLocalNode2ElemLocalNodeIndexMap<mNumSpatialDims> tCreateFaceLocalNode2ElemLocalNodeIndexMap;

        // get sideset faces
        auto tFaceLocalOrdinals = Plato::side_set_face_ordinals(mSpatialDomain.MeshSets, mSideSetName);
        auto tNumFaces = tFaceLocalOrdinals.size();
        Plato::ScalarArray3DT<ConfigT> tJacobians("jacobian", tNumFaces, mNumSpatialDimsOnFace, mNumSpatialDims);

        // set previous pressure at Gauss points container
        auto tNumCells = mSpatialDomain.Mesh.nelems();
        Plato::ScalarVectorT<PrevVelT> tPrevVelGP("previous velocity at Gauss point", tNumCells, mNumDofsPerNode);


        // set input state worksets
        auto tConfigWS       = aWorkSets.configuration();
        auto tPrevVelWS      = aWorkSets.previousVelocity();
        auto tCubatureWeight = mCubatureRule.getCubWeight();
        auto tBasisFunctions = mCubatureRule.getBasisFunctions();
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumFaces), LAMBDA_EXPRESSION(const Plato::OrdinalType & aFaceI)
        {

          auto tFaceOrdinal = tFaceLocalOrdinals[aFaceI];
          // for each element that the face is connected to: (either 1 or 2 elements)
          for( Plato::OrdinalType tElem = tFace2Elems_map[tFaceOrdinal]; tElem < tFace2Elems_map[tFaceOrdinal + 1]; tElem++ )
          {
              // create a map from face local node index to elem local node index
              Plato::OrdinalType tLocalNodeOrd[mNumSpatialDims];
              auto tCellOrdinal = tFace2Elems_elems[tElem];
              tCreateFaceLocalNode2ElemLocalNodeIndexMap(tCellOrdinal, tFaceOrdinal, tCell2Verts, tFace2Verts, tLocalNodeOrd);

              // calculate surface jacobians
              ConfigT tWeight(0.0);
              tComputeSurfaceJacobians(tCellOrdinal, aFaceI, tLocalNodeOrd, tConfigWS, tJacobians);
              tComputeSurfaceIntegralWeight(aFaceI, tCubatureWeight, tJacobians, tWeight);

              // compute unit normal vector
              auto tElemFaceOrdinal = Plato::get_face_ordinal<mNumSpatialDims>(tCellOrdinal, tFaceOrdinal, tElem2Faces);
              auto tUnitNormalVec = Plato::unit_normal_vector(tCellOrdinal, tElemFaceOrdinal, tCoords);

              // project into aResult workset
              tIntrplVectorField(tCellOrdinal, tBasisFunctions, tPrevVelWS, tPrevVelGP);
              for( Plato::OrdinalType tNode = 0; tNode < mNumNodesPerFace; tNode++ )
              {
                  for( Plato::OrdinalType tDof = 0; tDof < mNumDofsPerNode; tDof++ )
                  {
                      auto tCellDofOrdinal = (tLocalNodeOrd[tNode] * mNumDofsPerNode) + tDof;
                      aResult(tCellOrdinal, tCellDofOrdinal) += aMultiplier * tBasisFunctions(tNode) *
                          tUnitNormalVec(tDof) * tPrevVelGP(tCellOrdinal, tDof) * tWeight;
                  }
              }
          }
        }, "calculate surface momentum integral");
    }
};
// class MomentumSurfaceForces



template<typename PhysicsT, typename EvaluationT>
class PressureIncrementResidual :
    public Plato::FluidMechanics::AbstractVectorFunction<PhysicsT, EvaluationT>
{
private:
    static constexpr auto mNumDofsPerNode = PhysicsT::mNumDofsPerNode; /*!< number of degrees of freedom per node */
    static constexpr auto mNumDofsPerCell = PhysicsT::mNumDofsPerCell; /*!< number of degrees of freedom per cell */

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
    using PredictorT = typename EvaluationT::MomentumPredictorScalarType;

    Plato::DataMap& mDataMap;                   /*!< output database */
    const Plato::SpatialDomain& mSpatialDomain; /*!< Plato spatial model */
    Plato::LinearTetCubRuleDegreeOne<mNumSpatialDims> mCubatureRule; /*!< integration rule */

    Plato::Scalar mThetaOne = 0.5;
    Plato::Scalar mThetaTwo = 0.0;

    using MomentumForces = Plato::FluidMechanics::MomentumSurfaceForces<PhysicsT, EvaluationT>;
    std::unordered_map<std::string, std::shared_ptr<MomentumForces>> mMomentumNaturalBCs;

    using PrescribedForces = Plato::NaturalBC<mNumSpatialDims, mNumDofsPerNode>;
    std::unordered_map<std::string, std::shared_ptr<PrescribedForces>> mPrescribedNaturalBCs;

    using StateWorkSets = Plato::FluidMechanics::WorkSets<PhysicsT, EvaluationT>;

public:
    PressureIncrementResidual
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
            mThetaTwo = tTimeIntegration.get<Plato::Scalar>("Time Step Multiplier Theta 1", 0.5);
            mThetaTwo = tTimeIntegration.get<Plato::Scalar>("Time Step Multiplier Theta 2", 0.0);
        }
    }

    virtual ~PressureIncrementResidual(){}

    void evaluate
    (const StateWorkSets                & aWorkSets,
     Plato::ScalarMultiVectorT<ResultT> & aResult) const
    {
        auto tNumCells = aResult.extent(0);

        // set local data
        Plato::ScalarVectorT<ConfigT>    tCellVolume("cell weight", tNumCells);
        Plato::ScalarVectorT<CurPressT>  tCurPressGP("current pressure at Gauss point", tNumCells);
        Plato::ScalarVectorT<PrevPressT> tPrevPressGP("previous pressure at Gauss point", tNumCells);

        Plato::ScalarArray3DT<ConfigT> tGradient("cell gradient", tNumCells, mNumNodesPerCell, mNumSpatialDims);

        Plato::ScalarMultiVectorT<ResultT>    tIntForce("internal force at Gauss point", tNumCells, mNumSpatialDims);
        Plato::ScalarMultiVectorT<PrevVelT>   tPrevVelGP("previous velocity at Gauss point", tNumCells, mNumSpatialDims);
        Plato::ScalarMultiVectorT<PredictorT> tPredictorGP("predictor at Gauss point", tNumCells, mNumSpatialDims);

        // set local functors
        Plato::ComputeGradientWorkset<mNumSpatialDims> tComputeGradient;
        Plato::InterpolateFromNodal<mNumSpatialDims, mNumVelDofsPerNode,   0/*offset*/, mNumSpatialDims> tIntrplVectorField;
        Plato::InterpolateFromNodal<mNumSpatialDims, mNumPressDofsPerNode, 0/*offset*/, mNumSpatialDims> tIntrplScalarField;

        // set input state worksets
        auto tConfigWS    = aWorkSets.configuration();
        auto tPrevVelWS   = aWorkSets.previousVelocity();
        auto tCurPressWS  = aWorkSets.currentPressure();
        auto tTimeStepWS  = aWorkSets.timeStep();
        auto tPrevPressWS = aWorkSets.previousPressure();
        auto tPredictorWS = aWorkSets.predictor();
        auto tACompressWS = aWorkSets.artificialCompress();

        // transfer member data to device
        auto tThetaOne = mThetaOne;
        auto tThetaTwo = mThetaTwo;

        auto tCubWeight = mCubatureRule->getCubWeight();
        auto tBasisFunctions = mCubatureRule->getBasisFunctions();
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType &aCellOrdinal)
        {
            tComputeGradient(aCellOrdinal, tGradient, tConfigWS, tCellVolume);
            tCellVolume(aCellOrdinal) *= tCubWeight;

            // integrate previous advective force, which is defined as
            // \int_{\Omega_e}\frac{\partial N_p^a}{partial x_i}u_i^{n-1} d\Omega_e
            tIntrplVectorField(aCellOrdinal, tBasisFunctions, tPrevVelWS, tPrevVelGP);
            for(Plato::OrdinalType tNode = 0; tNode < mNumNodesPerCell; tNode++)
            {
                for(Plato::OrdinalType tDim = 0; tDim < mNumSpatialDims; tDim++)
                {
                    aResult(aCellOrdinal, tNode) += tCellVolume(aCellOrdinal) * tBasisFunctions(tNode) *
                        tGradient(aCellOrdinal, tNode, tDim) * tPrevVelGP(aCellOrdinal, tDim);
                }
            }

            // integrate current predicted advective force, which is defined as
            // \int_{\Omega_e}\frac{\partial N_p^a}{partial x_i} (u^{\ast}_i - u^{n-1}_i )^{n} d\Omega_e
            tIntrplVectorField(aCellOrdinal, tBasisFunctions, tPredictorWS, tPredictorGP);
            for(Plato::OrdinalType tNode = 0; tNode < mNumNodesPerCell; tNode++)
            {
                for(Plato::OrdinalType tDim = 0; tDim < mNumSpatialDims; tDim++)
                {
                    aResult(aCellOrdinal, tNode) += tThetaOne * tCellVolume(aCellOrdinal) * tBasisFunctions(tNode) *
                        tGradient(aCellOrdinal, tNode, tDim) * ( tPredictorGP(aCellOrdinal, tDim) - tPrevVelGP(aCellOrdinal, tDim) );
                }
            }

            // integrate continuity enforcement, which is defined as
            // -\Delta{t}\int_{\Omega_e}\frac{\partial N_p^a}{partial x_i}\frac{\partial p^{n+\theta_2}}{\partial x_i} d\Omega_e,
            // where
            // p^{n+\theta_2} = p^{n-1} + \theta_2*(p^{n} - p^{n-1})
            tIntrplScalarField(aCellOrdinal, tBasisFunctions, tCurPressWS , tCurPressGP);
            tIntrplScalarField(aCellOrdinal, tBasisFunctions, tPrevPressWS, tPrevPressGP);
            for(Plato::OrdinalType tNode = 0; tNode < mNumNodesPerCell; tNode++)
            {
                for(Plato::OrdinalType tDim = 0; tDim < mNumSpatialDims; tDim++)
                {
                    aResult(aCellOrdinal, tNode) -= tThetaOne * tCellVolume(aCellOrdinal) * tBasisFunctions(tNode) *
                        tGradient(aCellOrdinal, tNode, tDim) * ( tTimeStepWS(aCellOrdinal, tNode) *
                            tGradient(aCellOrdinal, tNode, tDim) * tPrevPressGP(aCellOrdinal) );

                    aResult(aCellOrdinal, tNode) -= tThetaOne * tCellVolume(aCellOrdinal) * tBasisFunctions(tNode) *
                        tGradient(aCellOrdinal, tNode, tDim) * ( tTimeStepWS(aCellOrdinal, tNode) * tThetaTwo *
                            tGradient(aCellOrdinal, tNode, tDim) * ( tCurPressGP(aCellOrdinal) - tPrevPressGP(aCellOrdinal) ) );
                }
            }

            // apply time step multiplier to internal force plus stabilizing force vectors,
            // i.e. F = \Delta{t} * \left( F_i^{int} + F_i^{stab} \right)
            for(Plato::OrdinalType tNode = 0; tNode < mNumNodesPerCell; tNode++)
            {
                auto tConstant = static_cast<Plato::Scalar>(-1) * tTimeStepWS(aCellOrdinal, tNode);
                aResult(aCellOrdinal, tNode) *= tConstant;
            }

            // integrate inertial forces, which are defined as
            // \int_{\Omega_e} N_p^a\left(\frac{1}{\beta^2}\right)(p^n - p^{n-1}) d\Omega_e
            for(Plato::OrdinalType tNode = 0; tNode < mNumNodesPerCell; tNode++)
            {
                auto tArtificialCompressibility = static_cast<Plato::Scalar>(1) / tACompressWS(aCellOrdinal, tNode);
                aResult(aCellOrdinal, tNode) += tCellVolume(aCellOrdinal) * tBasisFunctions(tNode) *
                    tArtificialCompressibility * ( tCurPressGP(aCellOrdinal) - tPrevPressGP(aCellOrdinal) );
            }

        }, "conservation of mass internal forces");
    }

    // todo: verify implementation with formulation
    void evaluateBoundary
    (const StateWorkSets                & aWorkSets,
     Plato::ScalarMultiVectorT<ResultT> & aResult) const
    {
        // calculate previous momentum forces, which are defined as
        // -\theta_1\Delta{t} \int_{\Gamma_u} N_u^a\left( -u_i^{n-1}n_i \right) d\Gamma
        auto tNumCells = aResult.extent(0);
        Plato::ScalarMultiVectorT<ResultT> tResultWS("previous momentum forces", tNumCells, mNumDofsPerCell);
        for(auto& tPair : mMomentumNaturalBCs)
        {
            tPair.second->operator()(aWorkSets, tResultWS, -1.0);
        }

        // multiply force vector by the corresponding nodal time steps
        auto tThetaOne = mThetaOne;
        auto tTimeStepWS = aWorkSets.timeStep();
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
        {
            for(Plato::OrdinalType tDof = 0; tDof < mNumPressDofsPerCell; tDof++)
            {
                aResult(aCellOrdinal, tDof) += tTimeStepWS(aCellOrdinal, tDof) * tThetaOne *
                    static_cast<Plato::Scalar>(-1.0) * tResultWS(aCellOrdinal, tDof);
            }
        }, "previous momentum forces");
    }

    void evaluatePrescribed
    (const StateWorkSets                & aWorkSets,
     Plato::ScalarMultiVectorT<ResultT> & aResult) const
    {
        // set input worksets
        auto tConfigWS  = aWorkSets.configuration();
        auto tControlWS = aWorkSets.control();
        auto tPrevVelWS = aWorkSets.previousVelocity();

        // calculate prescribed momentum forces, which are defined as
        // -\theta_1\Delta{t} \int_{\Gamma_u} N_u^a\left(u_i^{0}n_i\right) d\Gamma
        auto tNumCells = aResult.extent(0);
        Plato::ScalarMultiVectorT<ResultT> tResultWS("prescribed momentum forces", tNumCells, mNumDofsPerCell);
        for(auto& tPair : mPrescribedNaturalBCs)
        {
            tPair.second->get( mSpatialDomain, tPrevVelWS, tControlWS, tConfigWS, tResultWS);
        }

        // multiply force vector by the corresponding nodal time steps
        auto tThetaOne = mThetaOne;
        auto tTimeStepWS = aWorkSets.timeStep();
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
        {
            for(Plato::OrdinalType tDof = 0; tDof < mNumPressDofsPerCell; tDof++)
            {
                aResult(aCellOrdinal, tDof) += tTimeStepWS(aCellOrdinal, tDof) * tThetaOne *
                    static_cast<Plato::Scalar>(-1.0) * tResultWS(aCellOrdinal, tDof);
            }
        }, "prescribed momentum forces");
    }

private:
    void readNaturalBoundaryConditions(Teuchos::ParameterList& aInputs)
    {
        if(aInputs.isSublist("Pressure Increment Natural Boundary Conditions"))
        {
            auto tSublist = aInputs.sublist("Pressure Increment Natural Boundary Conditions");

            for (Teuchos::ParameterList::ConstIterator tItr = tSublist.begin(); tItr != tSublist.end(); ++tItr)
            {
                const Teuchos::ParameterEntry &tEntry = tSublist.entry(tItr);
                if (!tEntry.isList())
                {
                    THROWERR("Get Side Set Names: Parameter list block is not valid.  Expect lists only.")
                }

                const std::string &tParamListName = tSublist.name(tItr);
                if(tSublist.isSublist(tParamListName) == false)
                {
                    THROWERR(std::string("Parameter sublist with name '") + tParamListName.c_str() + "' is not defined.")
                }
                Teuchos::ParameterList &tParamList = tSublist.sublist(tParamListName);

                if(tSublist.isParameter("Sides") == false)
                {
                    THROWERR(std::string("Keyword 'Sides' is not define in Parameter Sublist with name '") + tParamListName + "'.")
                }
                const auto tSideSetName = tSublist.get<std::string>("Sides");

                auto tPrescribedBC = std::make_shared<PrescribedForces>(tParamListName, tParamList);
                mPrescribedNaturalBCs.insert(std::make_pair<std::string, std::shared_ptr<PrescribedForces>>(tSideSetName, tPrescribedBC));

                auto tMomentumBC = std::make_shared<MomentumForces>(mSpatialDomain, tSideSetName);
                mMomentumNaturalBCs.insert(std::make_pair<std::string, std::shared_ptr<MomentumForces>>(tSideSetName, tMomentumBC));
            }
        }
    }
};
// class PressureIncrementResidual




// todo: vector function
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
    static constexpr auto mNumDofsPerNode = PhysicsT::mNumDofsPerNode; /*!< number of degrees of freedom per node */
    static constexpr auto mNumDofsPerCell = PhysicsT::mNumDofsPerCell; /*!< number of degrees of freedom per cell */

    static constexpr auto mNumSpatialDims       = PhysicsT::SimplexT::mNumSpatialDims;         /*!< number of spatial dimensions */
    static constexpr auto mNumNodesPerCell      = PhysicsT::SimplexT::mNumNodesPerCell;        /*!< number of nodes per cell */
    static constexpr auto mNumControlsPerNode   = PhysicsT::SimplexT::mNumControl;             /*!< number of design variable fields */
    static constexpr auto mNumControlsPerCell   = mNumControlsPerNode * mNumNodesPerCell;       /*!< number of design variable fields */
    static constexpr auto mNumPressDofsPerCell  = PhysicsT::SimplexT::mNumMassDofsPerCell;     /*!< number of mass dofs per cell */
    static constexpr auto mNumTempDofsPerCell   = PhysicsT::SimplexT::mNumEnergyDofsPerCell;   /*!< number of energy dofs per cell */
    static constexpr auto mNumVelDofsPerCell    = PhysicsT::SimplexT::mNumMomentumDofsPerCell; /*!< number of momentum dofs per cell */
    static constexpr auto mNumPressDofsPerNode  = PhysicsT::SimplexT::mNumMassDofsPerNode;     /*!< number of mass dofs per node */
    static constexpr auto mNumTempDofsPerNode   = PhysicsT::SimplexT::mNumEnergyDofsPerNode;   /*!< number of energy dofs per node */
    static constexpr auto mNumVelDofsPerNode    = PhysicsT::SimplexT::mNumMomentumDofsPerNode; /*!< number of momentum dofs per node */

    static constexpr auto mNumConfigDofsPerNode = mNumSpatialDims;                    /*!< number of configuration degrees of freedom per cell */
    static constexpr auto mNumConfigDofsPerCell = mNumSpatialDims * mNumNodesPerCell; /*!< number of configuration degrees of freedom per cell */

    static constexpr auto mNumTimeStepsDofsPerNode = 1; /*!< number of time step dofs per node */
    static constexpr auto mNumACompressDofsPerNode = 1; /*!< number of artificial compressibility dofs per node */

    // forward automatic differentiation evaluation types
    using ResidualEvalT      = typename Plato::FluidMechanics::Evaluation<typename PhysicsT::SimplexT>::Residual;
    using GradConfigEvalT    = typename Plato::FluidMechanics::Evaluation<typename PhysicsT::SimplexT>::GradConfig;
    using GradControlEvalT   = typename Plato::FluidMechanics::Evaluation<typename PhysicsT::SimplexT>::GradControl;
    using GradCurVelEvalT    = typename Plato::FluidMechanics::Evaluation<typename PhysicsT::SimplexT>::GradCurMomentum;
    using GradPrevVelEvalT   = typename Plato::FluidMechanics::Evaluation<typename PhysicsT::SimplexT>::GradPrevMomentum;
    using GradCurTempEvalT   = typename Plato::FluidMechanics::Evaluation<typename PhysicsT::SimplexT>::GradCurEnergy;
    using GradPrevTempEvalT  = typename Plato::FluidMechanics::Evaluation<typename PhysicsT::SimplexT>::GradPrevEnergy;
    using GradCurPressEvalT  = typename Plato::FluidMechanics::Evaluation<typename PhysicsT::SimplexT>::GradCurMass;
    using GradPrevPressEvalT = typename Plato::FluidMechanics::Evaluation<typename PhysicsT::SimplexT>::GradPrevMass;
    using GradPredictorEvalT = typename Plato::FluidMechanics::Evaluation<typename PhysicsT::SimplexT>::GradPredictor;

    // element residual vector function types
    using ResidualFuncT      = std::shared_ptr<Plato::FluidMechanics::AbstractVectorFunction<typename PhysicsT::SimplexT, ResidualEvalT>>;
    using GradConfigFuncT    = std::shared_ptr<Plato::FluidMechanics::AbstractVectorFunction<typename PhysicsT::SimplexT, GradConfigEvalT>>;
    using GradControlFuncT   = std::shared_ptr<Plato::FluidMechanics::AbstractVectorFunction<typename PhysicsT::SimplexT, GradControlEvalT>>;
    using GradCurVelFuncT    = std::shared_ptr<Plato::FluidMechanics::AbstractVectorFunction<typename PhysicsT::SimplexT, GradCurVelEvalT>>;
    using GradPrevVelFuncT   = std::shared_ptr<Plato::FluidMechanics::AbstractVectorFunction<typename PhysicsT::SimplexT, GradPrevVelEvalT>>;
    using GradCurTempFuncT   = std::shared_ptr<Plato::FluidMechanics::AbstractVectorFunction<typename PhysicsT::SimplexT, GradCurTempEvalT>>;
    using GradPrevTempFuncT  = std::shared_ptr<Plato::FluidMechanics::AbstractVectorFunction<typename PhysicsT::SimplexT, GradPrevTempEvalT>>;
    using GradCurPressFuncT  = std::shared_ptr<Plato::FluidMechanics::AbstractVectorFunction<typename PhysicsT::SimplexT, GradCurPressEvalT>>;
    using GradPrevPressFuncT = std::shared_ptr<Plato::FluidMechanics::AbstractVectorFunction<typename PhysicsT::SimplexT, GradPrevPressEvalT>>;
    using GradPredictorFuncT = std::shared_ptr<Plato::FluidMechanics::AbstractVectorFunction<typename PhysicsT::SimplexT, GradPredictorEvalT>>;

    // element vector functions per element block, i.e. domain
    std::unordered_map<std::string, ResidualFuncT>      mResidualFuncs;
    std::unordered_map<std::string, GradConfigFuncT>    mGradConfigFuncs;
    std::unordered_map<std::string, GradControlFuncT>   mGradControlFuncs;
    std::unordered_map<std::string, GradCurVelFuncT>    mGradCurVelFuncs;
    std::unordered_map<std::string, GradPrevVelFuncT>   mGradPrevVelFuncs;
    std::unordered_map<std::string, GradCurTempFuncT>   mGradCurTempFuncs;
    std::unordered_map<std::string, GradPrevTempFuncT>  mGradPrevTempFuncs;
    std::unordered_map<std::string, GradCurPressFuncT>  mGradCurPressFuncs;
    std::unordered_map<std::string, GradPrevPressFuncT> mGradPrevPressFuncs;
    std::unordered_map<std::string, GradPredictorFuncT> mGradPredictorFuncs;

    Plato::DataMap& mDataMap;
    const Plato::SpatialModel& mSpatialModel;

    Plato::NodeCoordinate<mNumSpatialDims> mNodeCoordinate; /*!< node coordinates metadata */

    Plato::VectorEntryOrdinal<mNumSpatialDims, mNumSpatialDims>      mConfigEntryOrdinal;      /*!< configuration local-to-global ID map */
    Plato::VectorEntryOrdinal<mNumSpatialDims, mNumControlsPerNode>   mControlEntryOrdinal;     /*!< control local-to-global ID map */
    Plato::VectorEntryOrdinal<mNumSpatialDims, mNumVelDofsPerCell>   mVectorStateEntryOrdinal; /*!< vector state (e.g. velocity) local-to-global ID map */
    Plato::VectorEntryOrdinal<mNumSpatialDims, mNumPressDofsPerCell> mScalarStateEntryOrdinal; /*!< scalar state (e.g. pressure) local-to-global ID map */

    // define local type names
    using PrimalStates          = Plato::FluidMechanics::States;
    using ResidualWorkSets      = Plato::FluidMechanics::WorkSets<PhysicsT, ResidualEvalT>;
    using GradConfigWorkSets    = Plato::FluidMechanics::WorkSets<PhysicsT, GradConfigEvalT>;
    using GradControlWorkSets   = Plato::FluidMechanics::WorkSets<PhysicsT, GradControlEvalT>;
    using GradPredictorWorkSets = Plato::FluidMechanics::WorkSets<PhysicsT, GradPredictorEvalT>;
    using GradPrevVelWorkSets   = Plato::FluidMechanics::WorkSets<PhysicsT, GradPrevVelEvalT>;
    using GradPrevPressWorkSets = Plato::FluidMechanics::WorkSets<PhysicsT, GradPrevPressEvalT>;
    using GradPrevTempWorkSets  = Plato::FluidMechanics::WorkSets<PhysicsT, GradPrevTempEvalT>;
    using GradCurVelWorkSets    = Plato::FluidMechanics::WorkSets<PhysicsT, GradCurVelEvalT>;
    using GradCurPressWorkSets  = Plato::FluidMechanics::WorkSets<PhysicsT, GradCurPressEvalT>;
    using GradCurTempWorkSets   = Plato::FluidMechanics::WorkSets<PhysicsT, GradCurTempEvalT>;

public:
    /**************************************************************************//**
    * \brief Constructor
    * \param [in] aSpatialModel struct that contains the mesh, meshsets, domains, etc.
    * \param [in] aDataMap      problem-specific data map
    * \param [in] aInputs       Teuchos parameter list with input data
    * \param [in] aProblemType  problem type
    ******************************************************************************/
    VectorFunction
    (const std::string            & aName,
     const Plato::SpatialModel    & aSpatialModel,
           Plato::DataMap         & aDataMap,
           Teuchos::ParameterList & aInputs) :
        mSpatialModel(aSpatialModel),
        mDataMap(aDataMap),
        mConfigEntryOrdinal(Plato::VectorEntryOrdinal<mNumSpatialDims, mNumSpatialDims>(&aSpatialModel.Mesh)),
        mControlEntryOrdinal(Plato::VectorEntryOrdinal<mNumSpatialDims, mNumControlsPerNode>(&aSpatialModel.Mesh)),
        mVectorStateEntryOrdinal(Plato::VectorEntryOrdinal<mNumSpatialDims, mNumVelDofsPerCell>(&aSpatialModel.Mesh)),
        mScalarStateEntryOrdinal(Plato::VectorEntryOrdinal<mNumSpatialDims, mNumPressDofsPerCell>(&aSpatialModel.Mesh))
    {
        this->initialize(aName, aDataMap, aInputs);
    }

    Plato::ScalarVector value
    (const Plato::ScalarVector & aControls,
     const PrimalStates        & aStates) const
    {
        using ResultScalarT = typename ResidualEvalT::ResultScalarType;

        auto tNumNodes = mSpatialModel.Mesh.nverts();
        auto tLength = tNumNodes * mNumVelDofsPerCell;
        Plato::ScalarVector tReturnValue("Assembled Residual", tLength);

        // evaluate internal forces
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            ResidualWorkSets tWorkSets(tNumCells);
            this->setValueWorkSets(tDomain, aControls, aStates, tWorkSets);


            // evaluate internal forces
            auto tName = tDomain.getDomainName();
            Plato::ScalarMultiVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells, mNumDofsPerCell);
            mResidualFuncs.at(tName)->evaluate(tWorkSets, tResultWS);
            Plato::assemble_residual<mNumNodesPerCell, mNumDofsPerNode>
                (tNumCells, mVectorStateEntryOrdinal, tResultWS, tReturnValue);
        }

        // evaluate boundary forces
        {
            auto tNumCells = mSpatialModel.Mesh.nelems();
            ResidualWorkSets tWorkSets(tNumCells);
            this->setValueWorkSets(aControls, aStates, tWorkSets);

            // evaluate prescribed forces
            Plato::ScalarMultiVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells, mNumDofsPerCell);
            mResidualFuncs.begin()->evaluatePrescribed(tWorkSets, tResultWS);
            Plato::assemble_residual<mNumNodesPerCell, mNumDofsPerNode>
                (tNumCells, mVectorStateEntryOrdinal, tResultWS, tReturnValue);

            // evaluate balancing forces
            Plato::blas2::fill(0.0, tResultWS);
            mResidualFuncs.begin()->evaluateBoundary(tWorkSets, tResultWS);
            Plato::assemble_residual<mNumNodesPerCell, mNumDofsPerNode>
                (tNumCells, mVectorStateEntryOrdinal, tResultWS, tReturnValue);
        }

        return tReturnValue;
    }

    Teuchos::RCP<Plato::CrsMatrixType>
    gradientConfig
    (const Plato::ScalarVector & aControls,
     const PrimalStates        & aStates) const
    {
        using ResultScalarT = typename GradConfigEvalT::ResultScalarType;

        // create return matrix
        auto tMesh = mSpatialModel.Mesh;
        Teuchos::RCP<Plato::CrsMatrixType> tJacobian =
            Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumConfigDofsPerNode, mNumDofsPerNode>(&tMesh);

        // evaluate internal forces
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            GradConfigWorkSets tWorkSets(tNumCells);
            this->setGradConfigWorkSets(tDomain, aControls, aStates, tWorkSets);

            // evaluate internal forces
            auto tName = tDomain.getDomainName();
            Plato::ScalarMultiVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells, mNumDofsPerCell);
            mGradConfigFuncs.at(tName)->evaluate(tWorkSets, tResultWS);
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumConfigDofsPerNode, mNumDofsPerNode> tJacEntryOrdinal(tJacobian, &tMesh);
            Plato::assemble_transpose_jacobian(tDomain, mNumDofsPerCell, mNumConfigDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobian->entries());
        }

        // evaluate boundary forces
        {
            auto tNumCells = mSpatialModel.Mesh.nelems();
            GradConfigWorkSets tWorkSets(tNumCells);
            this->setGradConfigWorkSets(aControls, aStates, tWorkSets);

            // evaluate prescribed forces
            Plato::ScalarMultiVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells, mNumDofsPerCell);
            mGradConfigFuncs.begin()->evaluatePrescribed(tWorkSets, tResultWS);
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumConfigDofsPerNode, mNumDofsPerNode> tJacEntryOrdinal(tJacobian, &tMesh);
            Plato::assemble_transpose_jacobian(tNumCells, mNumDofsPerCell, mNumConfigDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobian->entries());

            // evaluate balancing forces
            Plato::blas2::fill(0.0, tResultWS);
            mGradConfigFuncs.begin()->evaluateBoundary(tWorkSets, tResultWS);
            Plato::assemble_transpose_jacobian(tNumCells, mNumDofsPerCell, mNumConfigDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobian->entries());
        }

        return tJacobian;
    }

    Teuchos::RCP<Plato::CrsMatrixType>
    gradientControl
    (const Plato::ScalarVector & aControls,
     const PrimalStates        & aStates) const
    {
        using ResultScalarT = typename GradControlEvalT::ResultScalarType;

        // create return matrix
        auto tMesh = mSpatialModel.Mesh;
        Teuchos::RCP<Plato::CrsMatrixType> tJacobian =
            Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumControlsPerNode, mNumDofsPerNode>( &tMesh );

        // evaluate internal forces
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            GradControlWorkSets tWorkSets(tNumCells);
            this->setGradControlWorkSets(tDomain, aControls, aStates, tWorkSets);

            // evaluate internal forces
            auto tName = tDomain.getDomainName();
            Plato::ScalarMultiVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells, mNumDofsPerCell);
            mGradControlFuncs.at(tName)->evaluate(tWorkSets, tResultWS);
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumControlsPerNode, mNumDofsPerNode> tJacEntryOrdinal(tJacobian, &tMesh);
            Plato::assemble_transpose_jacobian(tDomain, mNumDofsPerCell, mNumControlsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobian->entries());
        }

        // evaluate boundary forces
        {
            auto tNumCells = mSpatialModel.Mesh.nelems();
            GradControlWorkSets tWorkSets(tNumCells);
            this->setGradControlWorkSets(aControls, aStates, tWorkSets);

            // evaluate prescribed forces
            Plato::ScalarMultiVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells, mNumDofsPerCell);
            mGradControlFuncs.begin()->evaluatePrescribed(tWorkSets, tResultWS);
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumControlsPerNode, mNumDofsPerNode> tJacEntryOrdinal(tJacobian, &tMesh);
            Plato::assemble_transpose_jacobian(tNumCells, mNumDofsPerCell, mNumControlsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobian->entries());

            // evaluate balancing forces
            Plato::blas2::fill(0.0, tResultWS);
            mGradControlFuncs.begin()->evaluateBoundary(tWorkSets, tResultWS);
            Plato::assemble_transpose_jacobian(tNumCells, mNumDofsPerCell, mNumControlsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobian->entries());
        }

        return tJacobian;
    }

    Teuchos::RCP<Plato::CrsMatrixType>
    gradientPredictor
    (const Plato::ScalarVector & aControls,
     const PrimalStates        & aStates) const
    {
        using ResultScalarT = typename GradPredictorEvalT::ResultScalarType;

        // create return matrix
        auto tMesh = mSpatialModel.Mesh;
        Teuchos::RCP<Plato::CrsMatrixType> tJacobian =
            Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumVelDofsPerNode, mNumDofsPerNode>( &tMesh );

        // evaluate internal forces
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            GradPredictorWorkSets tWorkSets(tNumCells);
            this->setGradPredictorWorkSets(tDomain, aControls, aStates, tWorkSets);

            // evaluate internal forces
            auto tName = tDomain.getDomainName();
            Plato::ScalarMultiVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells, mNumDofsPerCell);
            mGradPredictorFuncs.at(tName)->evaluate(tWorkSets, tResultWS);
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumVelDofsPerNode, mNumDofsPerNode> tJacEntryOrdinal(tJacobian, &tMesh);
            Plato::assemble_transpose_jacobian(tDomain, mNumDofsPerCell, mNumVelDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobian->entries());
        }

        // evaluate boundary forces
        {
            auto tNumCells = mSpatialModel.Mesh.nelems();
            GradPredictorWorkSets tWorkSets(tNumCells);
            this->setGradPredictorWorkSets(aControls, aStates, tWorkSets);

            // evaluate prescribed forces
            Plato::ScalarMultiVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells, mNumDofsPerCell);
            mGradPredictorFuncs.begin()->evaluatePrescribed(tWorkSets);
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumVelDofsPerNode, mNumDofsPerNode> tJacEntryOrdinal(tJacobian, &tMesh);
            Plato::assemble_transpose_jacobian(tNumCells, mNumDofsPerCell, mNumVelDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobian->entries());

            // evaluate balancing forces
            Plato::blas2::fill(0.0, tResultWS);
            mGradPredictorFuncs.begin()->evaluateBoundary(tWorkSets, tResultWS);
            Plato::assemble_transpose_jacobian(tNumCells, mNumDofsPerCell, mNumVelDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobian->entries());
        }

        return tJacobian;
    }

    Teuchos::RCP<Plato::CrsMatrixType>
    gradientPreviousVel
    (const Plato::ScalarVector & aControls,
     const PrimalStates        & aStates) const
    {
        using ResultScalarT = typename GradPrevVelEvalT::ResultScalarType;

        // create return matrix
        auto tMesh = mSpatialModel.Mesh;
        Teuchos::RCP<Plato::CrsMatrixType> tJacobian =
            Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumVelDofsPerNode, mNumDofsPerNode>( &tMesh );

        // evaluate internal forces
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            GradPrevVelWorkSets tWorkSets(tNumCells);
            this->setGradPrevVelWorkSets(tDomain, aControls, aStates, tWorkSets);

            // evaluate internal forces
            auto tName = tDomain.getDomainName();
            Plato::ScalarMultiVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells, mNumDofsPerCell);
            mGradPrevVelFuncs.at(tName)->evaluate(tWorkSets, tResultWS);
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumVelDofsPerNode, mNumDofsPerNode> tJacEntryOrdinal(tJacobian, &tMesh);
            Plato::assemble_transpose_jacobian(tDomain, mNumDofsPerCell, mNumVelDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobian->entries());
        }

        // evaluate boundary forces
        {
            auto tNumCells = mSpatialModel.Mesh.nelems();
            GradPrevVelWorkSets tWorkSets(tNumCells);
            this->setGradPrevVelWorkSets(aControls, aStates, tWorkSets);

            // evaluate prescribed forces
            Plato::ScalarMultiVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells, mNumDofsPerCell);
            mGradPrevVelFuncs.begin()->evaluatePrescribed(tWorkSets, tResultWS);
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumVelDofsPerNode, mNumDofsPerNode> tJacEntryOrdinal(tJacobian, &tMesh);
            Plato::assemble_transpose_jacobian(tNumCells, mNumDofsPerCell, mNumVelDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobian->entries());

            // evaluate balancing forces
            Plato::blas2::fill(0.0, tResultWS);
            mGradPrevVelFuncs.begin()->evaluateBoundary(tWorkSets, tResultWS);
            Plato::assemble_transpose_jacobian(tNumCells, mNumDofsPerCell, mNumVelDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobian->entries());
        }

        return tJacobian;
    }

    Teuchos::RCP<Plato::CrsMatrixType>
    gradientPreviousPress
    (const Plato::ScalarVector & aControls,
     const PrimalStates        & aStates) const
    {
        using ResultScalarT = typename GradPrevPressEvalT::ResultScalarType;

        // create return matrix
        auto tMesh = mSpatialModel.Mesh;
        Teuchos::RCP<Plato::CrsMatrixType> tJacobian =
            Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumPressDofsPerNode, mNumDofsPerNode>( &tMesh );

        // evaluate internal forces
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            GradPrevPressWorkSets tWorkSets(tNumCells);
            this->setGradPrevPressWorkSets(tDomain, aControls, aStates, tWorkSets);

            // evaluate internal forces
            auto tName = tDomain.getDomainName();
            Plato::ScalarMultiVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells, mNumDofsPerCell);
            mGradPrevPressFuncs.at(tName)->evaluate(tWorkSets, tResultWS);
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumPressDofsPerNode, mNumDofsPerNode> tJacEntryOrdinal(tJacobian, &tMesh);
            Plato::assemble_transpose_jacobian(tDomain, mNumDofsPerCell, mNumPressDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobian->entries());
        }

        // evaluate boundary forces
        {
            auto tNumCells = mSpatialModel.Mesh.nelems();
            GradPrevPressWorkSets tWorkSets(tNumCells);
            this->setGradPrevPressWorkSets(aControls, aStates, tWorkSets);

            // evaluate prescribed forces
            Plato::ScalarMultiVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells, mNumDofsPerCell);
            mGradPrevPressFuncs.begin()->evaluatePrescribed(tWorkSets, tResultWS);
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumPressDofsPerNode, mNumDofsPerNode> tJacEntryOrdinal(tJacobian, &tMesh);
            Plato::assemble_transpose_jacobian(tNumCells, mNumDofsPerCell, mNumPressDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobian->entries());

            // evaluate balancing forces
            Plato::blas2::fill(0.0, tResultWS);
            mGradPrevPressFuncs.begin()->evaluateBoundary(tWorkSets, tResultWS);
            Plato::assemble_transpose_jacobian(tNumCells, mNumDofsPerCell, mNumPressDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobian->entries());
        }

        return tJacobian;
    }

    Teuchos::RCP<Plato::CrsMatrixType>
    gradientPreviousTemp
    (const Plato::ScalarVector & aControls,
     const PrimalStates        & aStates) const
    {
        using ResultScalarT = typename GradPrevTempEvalT::ResultScalarType;

        // create return matrix
        auto tMesh = mSpatialModel.Mesh;
        Teuchos::RCP<Plato::CrsMatrixType> tJacobian =
            Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumTempDofsPerNode, mNumDofsPerNode>( &tMesh );

        // evaluate internal forces
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            GradPrevTempWorkSets tWorkSets(tNumCells);
            this->setGradPrevTempWorkSets(tDomain, aControls, aStates, tWorkSets);

            // evaluate internal forces
            auto tName = tDomain.getDomainName();
            Plato::ScalarMultiVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells, mNumDofsPerCell);
            mGradPrevTempFuncs.at(tName)->evaluate(tWorkSets, tResultWS);
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumTempDofsPerNode, mNumDofsPerNode> tJacEntryOrdinal(tJacobian, &tMesh);
            Plato::assemble_transpose_jacobian(tDomain, mNumDofsPerCell, mNumTempDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobian->entries());
        }

        // evaluate boundary forces
        {
            auto tNumCells = mSpatialModel.Mesh.nelems();
            GradPrevTempWorkSets tWorkSets(tNumCells);
            this->setGradPrevTempWorkSets(aControls, aStates, tWorkSets);

            // evaluate prescribed forces
            Plato::ScalarMultiVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells, mNumDofsPerCell);
            mGradPrevTempFuncs.begin()->evaluatePrescribed(tWorkSets, tResultWS);
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumTempDofsPerNode, mNumDofsPerNode> tJacEntryOrdinal(tJacobian, &tMesh);
            Plato::assemble_transpose_jacobian(tNumCells, mNumDofsPerCell, mNumTempDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobian->entries());

            // evaluate balancing forces
            Plato::blas2::fill(0.0, tResultWS);
            mGradPrevTempFuncs.begin()->evaluateBoundary(tWorkSets, tResultWS);
            Plato::assemble_transpose_jacobian(tNumCells, mNumDofsPerCell, mNumTempDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobian->entries());
        }

        return tJacobian;
    }

    Teuchos::RCP<Plato::CrsMatrixType>
    gradientCurrentVel
    (const Plato::ScalarVector & aControls,
     const PrimalStates        & aStates) const
    {
        using ResultScalarT = typename GradCurVelEvalT::ResultScalarType;

        // create return matrix
        auto tMesh = mSpatialModel.Mesh;
        Teuchos::RCP<Plato::CrsMatrixType> tJacobian =
            Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumVelDofsPerNode, mNumDofsPerNode>( &tMesh );

        // evaluate internal forces
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            GradCurVelWorkSets tWorkSets(tNumCells);
            this->setGradCurVelWorkSets(tDomain, aControls, aStates, tWorkSets);

            // evaluate internal forces
            auto tName = tDomain.getDomainName();
            Plato::ScalarMultiVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells, mNumDofsPerCell);
            mGradCurVelFuncs.at(tName)->evaluate(tWorkSets, tResultWS);
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumVelDofsPerNode, mNumDofsPerNode> tJacEntryOrdinal(tJacobian, &tMesh);
            Plato::assemble_transpose_jacobian(tDomain, mNumDofsPerCell, mNumVelDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobian->entries());
        }

        // evaluate boundary forces
        {
            auto tNumCells = mSpatialModel.Mesh.nelems();
            GradCurVelWorkSets tWorkSets(tNumCells);
            this->setGradCurVelWorkSets(aControls, aStates, tWorkSets);

            // evaluate prescribed forces
            Plato::ScalarMultiVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells, mNumDofsPerCell);
            mGradCurVelFuncs.begin()->evaluatePrescribed(tWorkSets, tResultWS);
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumVelDofsPerNode, mNumDofsPerNode> tJacEntryOrdinal(tJacobian, &tMesh);
            Plato::assemble_transpose_jacobian(tNumCells, mNumDofsPerCell, mNumVelDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobian->entries());

            // evaluate balancing forces
            Plato::blas2::fill(0.0, tResultWS);
            mGradCurVelFuncs.begin()->evaluateBoundary(tWorkSets, tResultWS);
            Plato::assemble_transpose_jacobian(tNumCells, mNumDofsPerCell, mNumVelDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobian->entries());
        }

        return tJacobian;
    }

    Teuchos::RCP<Plato::CrsMatrixType>
    gradientCurrentPress
    (const Plato::ScalarVector & aControls,
     const PrimalStates        & aStates) const
    {
        using ResultScalarT = typename GradCurPressEvalT::ResultScalarType;

        // create return matrix
        auto tMesh = mSpatialModel.Mesh;
        Teuchos::RCP<Plato::CrsMatrixType> tJacobian =
            Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumPressDofsPerNode, mNumDofsPerNode>( &tMesh );

        // evaluate internal forces
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            GradCurPressWorkSets tWorkSets(tNumCells);
            this->setGradCurPressWorkSets(tDomain, aControls, aStates, tWorkSets);

            // evaluate internal forces
            auto tName = tDomain.getDomainName();
            Plato::ScalarMultiVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells, mNumDofsPerCell);
            mGradCurPressFuncs.at(tName)->evaluate(tWorkSets, tResultWS);
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumPressDofsPerNode, mNumDofsPerNode> tJacEntryOrdinal(tJacobian, &tMesh);
            Plato::assemble_transpose_jacobian(tDomain, mNumDofsPerCell, mNumPressDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobian->entries());
        }

        // evaluate boundary forces
        {
            auto tNumCells = mSpatialModel.Mesh.nelems();
            GradCurPressWorkSets tWorkSets(tNumCells);
            this->setGradCurPressWorkSets(aControls, aStates, tWorkSets);

            // evaluate prescribed forces
            Plato::ScalarMultiVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells, mNumDofsPerCell);
            mGradCurPressFuncs.begin()->evaluatePrescribed(tWorkSets, tResultWS);
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumPressDofsPerNode, mNumDofsPerNode> tJacEntryOrdinal(tJacobian, &tMesh);
            Plato::assemble_transpose_jacobian(tNumCells, mNumDofsPerCell, mNumPressDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobian->entries());

            // evaluate balancing forces
            Plato::blas2::fill(0.0, tResultWS);
            mGradCurPressFuncs.begin()->evaluateBoundary(tWorkSets, tResultWS);
            Plato::assemble_transpose_jacobian(tNumCells, mNumDofsPerCell, mNumPressDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobian->entries());
        }

        return tJacobian;
    }

    Teuchos::RCP<Plato::CrsMatrixType>
    gradientCurrentTemp
    (const Plato::ScalarVector & aControls,
     const PrimalStates        & aStates) const
    {
        // create return matrix
        auto tMesh = mSpatialModel.Mesh;
        Teuchos::RCP<Plato::CrsMatrixType> tJacobian =
            Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumTempDofsPerNode, mNumDofsPerNode>( &tMesh );

        // evaluate internal forces
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            GradCurTempWorkSets tWorkSets(tNumCells);
            this->setGradCurTempWorkSets(tDomain, aControls, aStates, tWorkSets);

            // evaluate internal forces
            auto tName = tDomain.getDomainName();
            Plato::ScalarMultiVectorT<GradCurTempEvalT> tResultWS("Cells Results", tNumCells, mNumDofsPerCell);
            mGradCurTempFuncs.at(tName)->evaluate(tWorkSets, tResultWS);
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumTempDofsPerNode, mNumDofsPerNode> tJacEntryOrdinal(tJacobian, &tMesh);
            Plato::assemble_transpose_jacobian(tDomain, mNumDofsPerCell, mNumTempDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobian->entries());
        }

        // evaluate prescribed forces
        {
            auto tNumCells = mSpatialModel.Mesh.nelems();
            GradCurTempWorkSets tWorkSets(tNumCells);
            this->setGradCurTempWorkSets(aControls, aStates, tWorkSets);

            // evaluate boundary forces
            Plato::ScalarMultiVectorT<GradCurTempEvalT> tResultWS("Cells Results", tNumCells, mNumDofsPerCell);
            mGradCurTempFuncs.begin()->evaluatePrescribed(tWorkSets, tResultWS);
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumTempDofsPerNode, mNumDofsPerNode> tJacEntryOrdinal(tJacobian, &tMesh);
            Plato::assemble_transpose_jacobian(tNumCells, mNumDofsPerCell, mNumTempDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobian->entries());

            // evaluate balancing forces
            Plato::blas2::fill(0.0, tResultWS);
            mGradCurTempFuncs.begin()->evaluateBoundary(tWorkSets, tResultWS);
            Plato::assemble_transpose_jacobian(tNumCells, mNumDofsPerCell, mNumTempDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobian->entries());
        }

        return tJacobian;
    }

private:
    void initialize
    (const std::string      & aName,
     Plato::DataMap         & aDataMap,
     Teuchos::ParameterList & aInputs)
    {
        typename PhysicsT::FunctionFactory tVecFuncFactory;
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tName = tDomain.getDomainName();

            mResidualFuncs[tName]  = tVecFuncFactory.template createVectorFunction<ResidualEvalT>
                (aName, tDomain, aDataMap, aInputs);

            mGradControlFuncs[tName] = tVecFuncFactory.template createVectorFunction<GradControlEvalT>
                (aName, tDomain, aDataMap, aInputs);

            mGradConfigFuncs[tName] = tVecFuncFactory.template createVectorFunction<GradConfigEvalT>
                (aName, tDomain, aDataMap, aInputs);

            mGradCurPressFuncs[tName] = tVecFuncFactory.template createVectorFunction<GradCurPressEvalT>
                (aName, tDomain, aDataMap, aInputs);

            mGradPrevPressFuncs[tName] = tVecFuncFactory.template createVectorFunction<GradPrevPressEvalT>
                (aName, tDomain, aDataMap, aInputs);

            mGradCurTempFuncs[tName] = tVecFuncFactory.template createVectorFunction<GradCurTempEvalT>
                (aName, tDomain, aDataMap, aInputs);

            mGradPrevTempFuncs[tName] = tVecFuncFactory.template createVectorFunction<GradPrevTempEvalT>
                (aName, tDomain, aDataMap, aInputs);

            mGradCurVelFuncs[tName] = tVecFuncFactory.template createVectorFunction<GradCurVelEvalT>
                (aName, tDomain, aDataMap, aInputs);

            mGradPrevVelFuncs[tName] = tVecFuncFactory.template createVectorFunction<GradPrevVelEvalT>
                (aName, tDomain, aDataMap, aInputs);

            mGradPredictorFuncs[tName] = tVecFuncFactory.template createVectorFunction<GradPredictorEvalT>
                (aName, tDomain, aDataMap, aInputs);
        }
    }

    void setValueWorkSets
    (const Plato::SpatialDomain & aDomain,
     const Plato::ScalarVector  & aControls,
     const PrimalStates         & aState,
           ResidualWorkSets     & aWorkSets) const
    {
        Plato::workset_state_scalar_scalar<mNumVelDofsPerNode, mNumNodesPerCell>
            (aDomain, mVectorStateEntryOrdinal, aState.vector("current predictor"), aWorkSets.predictor());

        Plato::workset_state_scalar_scalar<mNumVelDofsPerNode, mNumNodesPerCell>
            (aDomain, mVectorStateEntryOrdinal, aState.vector("current velocity"), aWorkSets.currentVelocity());

        Plato::workset_state_scalar_scalar<mNumPressDofsPerNode, mNumNodesPerCell>
            (aDomain, mScalarStateEntryOrdinal, aState.vector("current pressure"), aWorkSets.currentPressure());

        Plato::workset_state_scalar_scalar<mNumTempDofsPerNode, mNumNodesPerCell>
            (aDomain, mScalarStateEntryOrdinal, aState.vector("current temperature"), aWorkSets.currentTemperature());

        Plato::workset_state_scalar_scalar<mNumVelDofsPerNode, mNumNodesPerCell>
            (aDomain, mVectorStateEntryOrdinal, aState.vector("previous velocity"), aWorkSets.previousVelocity());

        Plato::workset_state_scalar_scalar<mNumPressDofsPerNode, mNumNodesPerCell>
            (aDomain, mScalarStateEntryOrdinal, aState.vector("previous pressure"), aWorkSets.previousPressure());

        Plato::workset_state_scalar_scalar<mNumTempDofsPerNode, mNumNodesPerCell>
            (aDomain, mScalarStateEntryOrdinal, aState.vector("previous temperature"), aWorkSets.previousTemperature());

        Plato::workset_control_scalar_scalar<mNumNodesPerCell>
            (aDomain, mControlEntryOrdinal, aControls, aWorkSets.control());

        Plato::workset_config_scalar<mNumSpatialDims, mNumNodesPerCell>
            (aDomain, mNodeCoordinate, aWorkSets.configuration());

        Plato::workset_state_scalar_scalar<mNumTimeStepsDofsPerNode, mNumNodesPerCell>
            (aDomain, mScalarStateEntryOrdinal, aState.vector("time step"), aWorkSets.timeStep());

        if(!aState.empty("artificial compressibility"))
        {
            Plato::workset_state_scalar_scalar<mNumACompressDofsPerNode, mNumNodesPerCell>
                (aDomain, mScalarStateEntryOrdinal, aState.vector("artificial compressibility"), aWorkSets.artificialCompress());
        }
    }

    void setValueWorkSets
    (const Plato::ScalarVector & aControls,
     const PrimalStates        & aState,
           ResidualWorkSets    & aWorkSets) const
    {
        auto tNumCells = aWorkSets.numCells();
        Plato::workset_state_scalar_scalar<mNumVelDofsPerNode, mNumNodesPerCell>
            (tNumCells, mVectorStateEntryOrdinal, aState.vector("current predictor"), aWorkSets.predictor());

        Plato::workset_state_scalar_scalar<mNumVelDofsPerNode, mNumNodesPerCell>
            (tNumCells, mVectorStateEntryOrdinal, aState.vector("current velocity"), aWorkSets.currentVelocity());

        Plato::workset_state_scalar_scalar<mNumPressDofsPerNode, mNumNodesPerCell>
            (tNumCells, mScalarStateEntryOrdinal, aState.vector("current pressure"), aWorkSets.currentPressure());

        Plato::workset_state_scalar_scalar<mNumTempDofsPerNode, mNumNodesPerCell>
            (tNumCells, mScalarStateEntryOrdinal, aState.vector("current temperature"), aWorkSets.currentTemperature());

        Plato::workset_state_scalar_scalar<mNumVelDofsPerNode, mNumNodesPerCell>
            (tNumCells, mVectorStateEntryOrdinal, aState.vector("previous velocity"), aWorkSets.previousVelocity());

        Plato::workset_state_scalar_scalar<mNumPressDofsPerNode, mNumNodesPerCell>
            (tNumCells, mScalarStateEntryOrdinal, aState.vector("previous pressure"), aWorkSets.previousPressure());

        Plato::workset_state_scalar_scalar<mNumTempDofsPerNode, mNumNodesPerCell>
            (tNumCells, mScalarStateEntryOrdinal, aState.vector("previous temperature"), aWorkSets.previousTemperature());

        Plato::workset_control_scalar_scalar<mNumNodesPerCell>
            (tNumCells, mControlEntryOrdinal, aControls, aWorkSets.control());

        Plato::workset_config_scalar<mNumSpatialDims, mNumNodesPerCell>
            (tNumCells, mNodeCoordinate, aWorkSets.configuration());

        Plato::workset_state_scalar_scalar<mNumTimeStepsDofsPerNode, mNumNodesPerCell>
            (tNumCells, mScalarStateEntryOrdinal, aState.vector("time step"), aWorkSets.timeStep());

        if(aState.empty("artificial compressibility") == false)
        {
            Plato::workset_state_scalar_scalar<mNumACompressDofsPerNode, mNumNodesPerCell>
                (tNumCells, mScalarStateEntryOrdinal, aState.vector("artificial compressibility"), aWorkSets.artificialCompress());
        }
    }

    void setGradConfigWorkSets
    (const Plato::SpatialDomain & aDomain,
     const Plato::ScalarVector  & aControls,
     const PrimalStates         & aState,
           GradConfigWorkSets   & aWorkSets) const
    {
        Plato::workset_state_scalar_scalar<mNumVelDofsPerNode, mNumNodesPerCell>
            (aDomain, mVectorStateEntryOrdinal, aState.vector("current predictor"), aWorkSets.predictor());

        Plato::workset_state_scalar_scalar<mNumVelDofsPerNode, mNumNodesPerCell>
            (aDomain, mVectorStateEntryOrdinal, aState.vector("current velocity"), aWorkSets.currentVelocity());

        Plato::workset_state_scalar_scalar<mNumPressDofsPerNode, mNumNodesPerCell>
            (aDomain, mScalarStateEntryOrdinal, aState.vector("current pressure"), aWorkSets.currentPressure());

        Plato::workset_state_scalar_scalar<mNumTempDofsPerNode, mNumNodesPerCell>
            (aDomain, mScalarStateEntryOrdinal, aState.vector("current temperature"), aWorkSets.currentTemperature());

        Plato::workset_state_scalar_scalar<mNumVelDofsPerNode, mNumNodesPerCell>
            (aDomain, mVectorStateEntryOrdinal, aState.vector("previous velocity"), aWorkSets.previousVelocity());

        Plato::workset_state_scalar_scalar<mNumPressDofsPerNode, mNumNodesPerCell>
            (aDomain, mScalarStateEntryOrdinal, aState.vector("previous pressure"), aWorkSets.previousPressure());

        Plato::workset_state_scalar_scalar<mNumTempDofsPerNode, mNumNodesPerCell>
            (aDomain, mScalarStateEntryOrdinal, aState.vector("previous temperature"), aWorkSets.previousTemperature());

        Plato::workset_control_scalar_scalar<mNumNodesPerCell>
            (aDomain, mControlEntryOrdinal, aControls, aWorkSets.control());

        using ConfigScalarT = typename GradConfigEvalT::ConfigScalarType;
        Plato::workset_config_fad<mNumSpatialDims, mNumNodesPerCell, mNumConfigDofsPerCell, ConfigScalarT>
            (aDomain, mNodeCoordinate, aWorkSets.configuration());

        Plato::workset_state_scalar_scalar<mNumTimeStepsDofsPerNode, mNumNodesPerCell>
            (aDomain, mScalarStateEntryOrdinal, aState.vector("time step"), aWorkSets.timeStep());

        if(!aState.empty("artificial compressibility"))
        {
            Plato::workset_state_scalar_scalar<mNumACompressDofsPerNode, mNumNodesPerCell>
                (aDomain, mScalarStateEntryOrdinal, aState.vector("artificial compressibility"), aWorkSets.artificialCompress());
        }
    }

    void setGradConfigWorkSets
    (const Plato::ScalarVector & aControls,
     const PrimalStates        & aState,
           GradConfigWorkSets  & aWorkSets) const
    {
        auto tNumCells = aWorkSets.numCells();
        Plato::workset_state_scalar_scalar<mNumVelDofsPerNode, mNumNodesPerCell>
            (tNumCells, mVectorStateEntryOrdinal, aState.vector("current predictor"), aWorkSets.predictor());

        Plato::workset_state_scalar_scalar<mNumVelDofsPerNode, mNumNodesPerCell>
            (tNumCells, mVectorStateEntryOrdinal, aState.vector("current velocity"), aWorkSets.currentVelocity());

        Plato::workset_state_scalar_scalar<mNumPressDofsPerNode, mNumNodesPerCell>
            (tNumCells, mScalarStateEntryOrdinal, aState.vector("current pressure"), aWorkSets.currentPressure());

        Plato::workset_state_scalar_scalar<mNumTempDofsPerNode, mNumNodesPerCell>
            (tNumCells, mScalarStateEntryOrdinal, aState.vector("current temperature"), aWorkSets.currentTemperature());

        Plato::workset_state_scalar_scalar<mNumVelDofsPerNode, mNumNodesPerCell>
            (tNumCells, mVectorStateEntryOrdinal, aState.vector("previous velocity"), aWorkSets.previousVelocity());

        Plato::workset_state_scalar_scalar<mNumPressDofsPerNode, mNumNodesPerCell>
            (tNumCells, mScalarStateEntryOrdinal, aState.vector("previous pressure"), aWorkSets.previousPressure());

        Plato::workset_state_scalar_scalar<mNumTempDofsPerNode, mNumNodesPerCell>
            (tNumCells, mScalarStateEntryOrdinal, aState.vector("previous temperature"), aWorkSets.previousTemperature());

        Plato::workset_control_scalar_scalar<mNumNodesPerCell>
            (tNumCells, mControlEntryOrdinal, aControls, aWorkSets.control());

        using ConfigScalarT = typename GradConfigEvalT::ConfigScalarType;
        Plato::workset_config_fad<mNumSpatialDims, mNumNodesPerCell, mNumConfigDofsPerCell, ConfigScalarT>
            (tNumCells, mNodeCoordinate, aWorkSets.configuration());

        Plato::workset_state_scalar_scalar<mNumTimeStepsDofsPerNode, mNumNodesPerCell>
            (tNumCells, mScalarStateEntryOrdinal, aState.vector("time step"), aWorkSets.timeStep());

        if(aState.empty("artificial compressibility") == false)
        {
            Plato::workset_state_scalar_scalar<mNumACompressDofsPerNode, mNumNodesPerCell>
                (tNumCells, mScalarStateEntryOrdinal, aState.vector("artificial compressibility"), aWorkSets.artificialCompress());
        }
    }

    void setGradControlWorkSets
    (const Plato::SpatialDomain & aDomain,
     const Plato::ScalarVector  & aControls,
     const PrimalStates         & aState,
           GradControlWorkSets  & aWorkSets) const
    {
        Plato::workset_state_scalar_scalar<mNumVelDofsPerNode, mNumNodesPerCell>
            (aDomain, mVectorStateEntryOrdinal, aState.vector("current predictor"), aWorkSets.predictor());

        Plato::workset_state_scalar_scalar<mNumVelDofsPerNode, mNumNodesPerCell>
            (aDomain, mVectorStateEntryOrdinal, aState.vector("current velocity"), aWorkSets.currentVelocity());

        Plato::workset_state_scalar_scalar<mNumPressDofsPerNode, mNumNodesPerCell>
            (aDomain, mScalarStateEntryOrdinal, aState.vector("current pressure"), aWorkSets.currentPressure());

        Plato::workset_state_scalar_scalar<mNumTempDofsPerNode, mNumNodesPerCell>
            (aDomain, mScalarStateEntryOrdinal, aState.vector("current temperature"), aWorkSets.currentTemperature());

        Plato::workset_state_scalar_scalar<mNumVelDofsPerNode, mNumNodesPerCell>
            (aDomain, mVectorStateEntryOrdinal, aState.vector("previous velocity"), aWorkSets.previousVelocity());

        Plato::workset_state_scalar_scalar<mNumPressDofsPerNode, mNumNodesPerCell>
            (aDomain, mScalarStateEntryOrdinal, aState.vector("previous pressure"), aWorkSets.previousPressure());

        Plato::workset_state_scalar_scalar<mNumTempDofsPerNode, mNumNodesPerCell>
            (aDomain, mScalarStateEntryOrdinal, aState.vector("previous temperature"), aWorkSets.previousTemperature());

        using ControlScalarT = typename GradControlEvalT::ControlScalarType;
        Plato::workset_control_scalar_fad<mNumNodesPerCell, ControlScalarT>
            (aDomain, mControlEntryOrdinal, aControls, aWorkSets.control());

        Plato::workset_config_scalar<mNumSpatialDims, mNumNodesPerCell>
            (aDomain, mNodeCoordinate, aWorkSets.configuration());

        Plato::workset_state_scalar_scalar<mNumTimeStepsDofsPerNode, mNumNodesPerCell>
            (aDomain, mScalarStateEntryOrdinal, aState.vector("time step"), aWorkSets.timeStep());

        if(!aState.empty("artificial compressibility"))
        {
            Plato::workset_state_scalar_scalar<mNumACompressDofsPerNode, mNumNodesPerCell>
                (aDomain, mScalarStateEntryOrdinal, aState.vector("artificial compressibility"), aWorkSets.artificialCompress());
        }
    }

    void setGradControlWorkSets
    (const Plato::ScalarVector & aControls,
     const PrimalStates        & aState,
           GradControlWorkSets & aWorkSets) const
    {
        auto tNumCells = aWorkSets.numCells();
        Plato::workset_state_scalar_scalar<mNumVelDofsPerNode, mNumNodesPerCell>
            (tNumCells, mVectorStateEntryOrdinal, aState.vector("current predictor"), aWorkSets.predictor());

        Plato::workset_state_scalar_scalar<mNumVelDofsPerNode, mNumNodesPerCell>
            (tNumCells, mVectorStateEntryOrdinal, aState.vector("current velocity"), aWorkSets.currentVelocity());

        Plato::workset_state_scalar_scalar<mNumPressDofsPerNode, mNumNodesPerCell>
            (tNumCells, mScalarStateEntryOrdinal, aState.vector("current pressure"), aWorkSets.currentPressure());

        Plato::workset_state_scalar_scalar<mNumTempDofsPerNode, mNumNodesPerCell>
            (tNumCells, mScalarStateEntryOrdinal, aState.vector("current temperature"), aWorkSets.currentTemperature());

        Plato::workset_state_scalar_scalar<mNumVelDofsPerNode, mNumNodesPerCell>
            (tNumCells, mVectorStateEntryOrdinal, aState.vector("previous velocity"), aWorkSets.previousVelocity());

        Plato::workset_state_scalar_scalar<mNumPressDofsPerNode, mNumNodesPerCell>
            (tNumCells, mScalarStateEntryOrdinal, aState.vector("previous pressure"), aWorkSets.previousPressure());

        Plato::workset_state_scalar_scalar<mNumTempDofsPerNode, mNumNodesPerCell>
            (tNumCells, mScalarStateEntryOrdinal, aState.vector("previous temperature"), aWorkSets.previousTemperature());

        using ControlScalarT = typename GradControlEvalT::ControlScalarType;
        Plato::workset_control_scalar_fad<mNumNodesPerCell, ControlScalarT>
            (tNumCells, mControlEntryOrdinal, aControls, aWorkSets.control());

        Plato::workset_config_scalar<mNumSpatialDims, mNumNodesPerCell>
            (tNumCells, mNodeCoordinate, aWorkSets.configuration());

        Plato::workset_state_scalar_scalar<mNumTimeStepsDofsPerNode, mNumNodesPerCell>
            (tNumCells, mScalarStateEntryOrdinal, aState.vector("time step"), aWorkSets.timeStep());

        if(aState.empty("artificial compressibility") == false)
        {
            Plato::workset_state_scalar_scalar<mNumACompressDofsPerNode, mNumNodesPerCell>
                (tNumCells, mScalarStateEntryOrdinal, aState.vector("artificial compressibility"), aWorkSets.artificialCompress());
        }
    }

    void setGradPredictorWorkSets
    (const Plato::SpatialDomain  & aDomain,
     const Plato::ScalarVector   & aControls,
     const PrimalStates          & aState,
           GradPredictorWorkSets & aWorkSets) const
    {
        using PredictorScalarT = typename GradPredictorEvalT::MomentumPredictorScalarType;
        Plato::workset_state_scalar_fad<mNumVelDofsPerNode, mNumNodesPerCell, PredictorScalarT>
            (aDomain, mVectorStateEntryOrdinal, aState.vector("current predictor"), aWorkSets.predictor());

        Plato::workset_state_scalar_scalar<mNumVelDofsPerNode, mNumNodesPerCell>
            (aDomain, mVectorStateEntryOrdinal, aState.vector("current velocity"), aWorkSets.currentVelocity());

        Plato::workset_state_scalar_scalar<mNumPressDofsPerNode, mNumNodesPerCell>
            (aDomain, mScalarStateEntryOrdinal, aState.vector("current pressure"), aWorkSets.currentPressure());

        Plato::workset_state_scalar_scalar<mNumTempDofsPerNode, mNumNodesPerCell>
            (aDomain, mScalarStateEntryOrdinal, aState.vector("current temperature"), aWorkSets.currentTemperature());

        Plato::workset_state_scalar_scalar<mNumVelDofsPerNode, mNumNodesPerCell>
            (aDomain, mVectorStateEntryOrdinal, aState.vector("previous velocity"), aWorkSets.previousVelocity());

        Plato::workset_state_scalar_scalar<mNumPressDofsPerNode, mNumNodesPerCell>
            (aDomain, mScalarStateEntryOrdinal, aState.vector("previous pressure"), aWorkSets.previousPressure());

        Plato::workset_state_scalar_scalar<mNumTempDofsPerNode, mNumNodesPerCell>
            (aDomain, mScalarStateEntryOrdinal, aState.vector("previous temperature"), aWorkSets.previousTemperature());

        Plato::workset_control_scalar_scalar<mNumNodesPerCell>
            (aDomain, mControlEntryOrdinal, aControls, aWorkSets.control());

        Plato::workset_config_scalar<mNumSpatialDims, mNumNodesPerCell>
            (aDomain, mNodeCoordinate, aWorkSets.configuration());

        Plato::workset_state_scalar_scalar<mNumTimeStepsDofsPerNode, mNumNodesPerCell>
            (aDomain, mScalarStateEntryOrdinal, aState.vector("time step"), aWorkSets.timeStep());

        if(!aState.empty("artificial compressibility"))
        {
            Plato::workset_state_scalar_scalar<mNumACompressDofsPerNode, mNumNodesPerCell>
                (aDomain, mScalarStateEntryOrdinal, aState.vector("artificial compressibility"), aWorkSets.artificialCompress());
        }
    }

    void setGradPredictorWorkSets
    (const Plato::ScalarVector   & aControls,
     const PrimalStates          & aState,
           GradPredictorWorkSets & aWorkSets) const
    {
        auto tNumCells = aWorkSets.numCells();

        using PredictorScalarT = typename GradPredictorEvalT::MomentumPredictorScalarType;
        Plato::workset_state_scalar_fad<mNumVelDofsPerNode, mNumNodesPerCell, PredictorScalarT>
            (tNumCells, mVectorStateEntryOrdinal, aState.vector("current predictor"), aWorkSets.predictor());

        Plato::workset_state_scalar_scalar<mNumVelDofsPerNode, mNumNodesPerCell>
            (tNumCells, mVectorStateEntryOrdinal, aState.vector("current velocity"), aWorkSets.currentVelocity());

        Plato::workset_state_scalar_scalar<mNumPressDofsPerNode, mNumNodesPerCell>
            (tNumCells, mScalarStateEntryOrdinal, aState.vector("current pressure"), aWorkSets.currentPressure());

        Plato::workset_state_scalar_scalar<mNumTempDofsPerNode, mNumNodesPerCell>
            (tNumCells, mScalarStateEntryOrdinal, aState.vector("current temperature"), aWorkSets.currentTemperature());

        Plato::workset_state_scalar_scalar<mNumVelDofsPerNode, mNumNodesPerCell>
            (tNumCells, mVectorStateEntryOrdinal, aState.vector("previous velocity"), aWorkSets.previousVelocity());

        Plato::workset_state_scalar_scalar<mNumPressDofsPerNode, mNumNodesPerCell>
            (tNumCells, mScalarStateEntryOrdinal, aState.vector("previous pressure"), aWorkSets.previousPressure());

        Plato::workset_state_scalar_scalar<mNumTempDofsPerNode, mNumNodesPerCell>
            (tNumCells, mScalarStateEntryOrdinal, aState.vector("previous temperature"), aWorkSets.previousTemperature());

        Plato::workset_control_scalar_scalar<mNumNodesPerCell>
            (tNumCells, mControlEntryOrdinal, aControls, aWorkSets.control());

        Plato::workset_config_scalar<mNumSpatialDims, mNumNodesPerCell>
            (tNumCells, mNodeCoordinate, aWorkSets.configuration());

        Plato::workset_state_scalar_scalar<mNumTimeStepsDofsPerNode, mNumNodesPerCell>
            (tNumCells, mScalarStateEntryOrdinal, aState.vector("time step"), aWorkSets.timeStep());

        if(aState.empty("artificial compressibility") == false)
        {
            Plato::workset_state_scalar_scalar<mNumACompressDofsPerNode, mNumNodesPerCell>
                (tNumCells, mScalarStateEntryOrdinal, aState.vector("artificial compressibility"), aWorkSets.artificialCompress());
        }
    }

    void setGradPrevVelWorkSets
    (const Plato::SpatialDomain & aDomain,
     const Plato::ScalarVector  & aControls,
     const PrimalStates         & aState,
           GradPrevVelWorkSets  & aWorkSets) const
    {
        Plato::workset_state_scalar_scalar<mNumVelDofsPerNode, mNumNodesPerCell>
            (aDomain, mVectorStateEntryOrdinal, aState.vector("current predictor"), aWorkSets.predictor());

        Plato::workset_state_scalar_scalar<mNumVelDofsPerNode, mNumNodesPerCell>
            (aDomain, mVectorStateEntryOrdinal, aState.vector("current velocity"), aWorkSets.currentVelocity());

        Plato::workset_state_scalar_scalar<mNumPressDofsPerNode, mNumNodesPerCell>
            (aDomain, mScalarStateEntryOrdinal, aState.vector("current pressure"), aWorkSets.currentPressure());

        Plato::workset_state_scalar_scalar<mNumTempDofsPerNode, mNumNodesPerCell>
            (aDomain, mScalarStateEntryOrdinal, aState.vector("current temperature"), aWorkSets.currentTemperature());

        using PrevVelScalarT = typename GradPrevVelEvalT::PreviousMomentumScalarType;
        Plato::workset_state_scalar_fad<mNumVelDofsPerNode, mNumNodesPerCell, PrevVelScalarT>
            (aDomain, mVectorStateEntryOrdinal, aState.vector("previous velocity"), aWorkSets.previousVelocity());

        Plato::workset_state_scalar_scalar<mNumPressDofsPerNode, mNumNodesPerCell>
            (aDomain, mScalarStateEntryOrdinal, aState.vector("previous pressure"), aWorkSets.previousPressure());

        Plato::workset_state_scalar_scalar<mNumTempDofsPerNode, mNumNodesPerCell>
            (aDomain, mScalarStateEntryOrdinal, aState.vector("previous temperature"), aWorkSets.previousTemperature());

        Plato::workset_control_scalar_scalar<mNumNodesPerCell>
            (aDomain, mControlEntryOrdinal, aControls, aWorkSets.control());

        Plato::workset_config_scalar<mNumSpatialDims, mNumNodesPerCell>
            (aDomain, mNodeCoordinate, aWorkSets.configuration());

        Plato::workset_state_scalar_scalar<mNumTimeStepsDofsPerNode, mNumNodesPerCell>
            (aDomain, mScalarStateEntryOrdinal, aState.vector("time step"), aWorkSets.timeStep());

        if(!aState.empty("artificial compressibility"))
        {
            Plato::workset_state_scalar_scalar<mNumACompressDofsPerNode, mNumNodesPerCell>
                (aDomain, mScalarStateEntryOrdinal, aState.vector("artificial compressibility"), aWorkSets.artificialCompress());
        }
    }

    void setGradPrevVelWorkSets
    (const Plato::ScalarVector & aControls,
     const PrimalStates        & aState,
           GradPrevVelWorkSets & aWorkSets) const
    {
        auto tNumCells = aWorkSets.numCells();

        Plato::workset_state_scalar_scalar<mNumVelDofsPerNode, mNumNodesPerCell>
            (tNumCells, mVectorStateEntryOrdinal, aState.vector("current predictor"), aWorkSets.predictor());

        Plato::workset_state_scalar_scalar<mNumVelDofsPerNode, mNumNodesPerCell>
            (tNumCells, mVectorStateEntryOrdinal, aState.vector("current velocity"), aWorkSets.currentVelocity());

        Plato::workset_state_scalar_scalar<mNumPressDofsPerNode, mNumNodesPerCell>
            (tNumCells, mScalarStateEntryOrdinal, aState.vector("current pressure"), aWorkSets.currentPressure());

        Plato::workset_state_scalar_scalar<mNumTempDofsPerNode, mNumNodesPerCell>
            (tNumCells, mScalarStateEntryOrdinal, aState.vector("current temperature"), aWorkSets.currentTemperature());

        using PrevVelScalarT = typename GradPrevVelEvalT::PreviousMomentumScalarType;
        Plato::workset_state_scalar_fad<mNumVelDofsPerNode, mNumNodesPerCell, PrevVelScalarT>
            (tNumCells, mVectorStateEntryOrdinal, aState.vector("previous velocity"), aWorkSets.previousVelocity());

        Plato::workset_state_scalar_scalar<mNumPressDofsPerNode, mNumNodesPerCell>
            (tNumCells, mScalarStateEntryOrdinal, aState.vector("previous pressure"), aWorkSets.previousPressure());

        Plato::workset_state_scalar_scalar<mNumTempDofsPerNode, mNumNodesPerCell>
            (tNumCells, mScalarStateEntryOrdinal, aState.vector("previous temperature"), aWorkSets.previousTemperature());

        Plato::workset_control_scalar_scalar<mNumNodesPerCell>
            (tNumCells, mControlEntryOrdinal, aControls, aWorkSets.control());

        Plato::workset_config_scalar<mNumSpatialDims, mNumNodesPerCell>
            (tNumCells, mNodeCoordinate, aWorkSets.configuration());

        Plato::workset_state_scalar_scalar<mNumTimeStepsDofsPerNode, mNumNodesPerCell>
            (tNumCells, mScalarStateEntryOrdinal, aState.vector("time step"), aWorkSets.timeStep());

        if(aState.empty("artificial compressibility") == false)
        {
            Plato::workset_state_scalar_scalar<mNumACompressDofsPerNode, mNumNodesPerCell>
                (tNumCells, mScalarStateEntryOrdinal, aState.vector("artificial compressibility"), aWorkSets.artificialCompress());
        }
    }

    void setGradPrevPressWorkSets
    (const Plato::SpatialDomain  & aDomain,
     const Plato::ScalarVector   & aControls,
     const PrimalStates          & aState,
           GradPrevPressWorkSets & aWorkSets) const
    {
        Plato::workset_state_scalar_scalar<mNumVelDofsPerNode, mNumNodesPerCell>
            (aDomain, mVectorStateEntryOrdinal, aState.vector("current predictor"), aWorkSets.predictor());

        Plato::workset_state_scalar_scalar<mNumVelDofsPerNode, mNumNodesPerCell>
            (aDomain, mVectorStateEntryOrdinal, aState.vector("current velocity"), aWorkSets.currentVelocity());

        Plato::workset_state_scalar_scalar<mNumPressDofsPerNode, mNumNodesPerCell>
            (aDomain, mScalarStateEntryOrdinal, aState.vector("current pressure"), aWorkSets.currentPressure());

        Plato::workset_state_scalar_scalar<mNumTempDofsPerNode, mNumNodesPerCell>
            (aDomain, mScalarStateEntryOrdinal, aState.vector("current temperature"), aWorkSets.currentTemperature());

        Plato::workset_state_scalar_scalar<mNumVelDofsPerNode, mNumNodesPerCell>
            (aDomain, mVectorStateEntryOrdinal, aState.vector("previous velocity"), aWorkSets.previousVelocity());

        using PrevPressScalarT = typename GradPrevPressEvalT::PreviousMassScalarType;
        Plato::workset_state_scalar_fad<mNumPressDofsPerNode, mNumNodesPerCell, PrevPressScalarT>
            (aDomain, mScalarStateEntryOrdinal, aState.vector("previous pressure"), aWorkSets.previousPressure());

        Plato::workset_state_scalar_scalar<mNumTempDofsPerNode, mNumNodesPerCell>
            (aDomain, mScalarStateEntryOrdinal, aState.vector("previous temperature"), aWorkSets.previousTemperature());

        Plato::workset_control_scalar_scalar<mNumNodesPerCell>
            (aDomain, mControlEntryOrdinal, aControls, aWorkSets.control());

        Plato::workset_config_scalar<mNumSpatialDims, mNumNodesPerCell>
            (aDomain, mNodeCoordinate, aWorkSets.configuration());

        Plato::workset_state_scalar_scalar<mNumTimeStepsDofsPerNode, mNumNodesPerCell>
            (aDomain, mScalarStateEntryOrdinal, aState.vector("time step"), aWorkSets.timeStep());

        if(!aState.empty("artificial compressibility"))
        {
            Plato::workset_state_scalar_scalar<mNumACompressDofsPerNode, mNumNodesPerCell>
                (aDomain, mScalarStateEntryOrdinal, aState.vector("artificial compressibility"), aWorkSets.artificialCompress());
        }
    }

    void setGradPrevPressWorkSets
    (const Plato::ScalarVector   & aControls,
     const PrimalStates          & aState,
           GradPrevPressWorkSets & aWorkSets) const
    {
        auto tNumCells = aWorkSets.numCells();

        Plato::workset_state_scalar_scalar<mNumVelDofsPerNode, mNumNodesPerCell>
            (tNumCells, mVectorStateEntryOrdinal, aState.vector("current predictor"), aWorkSets.predictor());

        Plato::workset_state_scalar_scalar<mNumVelDofsPerNode, mNumNodesPerCell>
            (tNumCells, mVectorStateEntryOrdinal, aState.vector("current velocity"), aWorkSets.currentVelocity());

        Plato::workset_state_scalar_scalar<mNumPressDofsPerNode, mNumNodesPerCell>
            (tNumCells, mScalarStateEntryOrdinal, aState.vector("current pressure"), aWorkSets.currentPressure());

        Plato::workset_state_scalar_scalar<mNumTempDofsPerNode, mNumNodesPerCell>
            (tNumCells, mScalarStateEntryOrdinal, aState.vector("current temperature"), aWorkSets.currentTemperature());

        Plato::workset_state_scalar_scalar<mNumVelDofsPerNode, mNumNodesPerCell>
            (tNumCells, mVectorStateEntryOrdinal, aState.vector("previous velocity"), aWorkSets.previousVelocity());

        using PrevPressScalarT = typename GradPrevPressEvalT::PreviousMassScalarType;
        Plato::workset_state_scalar_fad<mNumPressDofsPerNode, mNumNodesPerCell, PrevPressScalarT>
            (tNumCells, mScalarStateEntryOrdinal, aState.vector("previous pressure"), aWorkSets.previousPressure());

        Plato::workset_state_scalar_scalar<mNumTempDofsPerNode, mNumNodesPerCell>
            (tNumCells, mScalarStateEntryOrdinal, aState.vector("previous temperature"), aWorkSets.previousTemperature());

        Plato::workset_control_scalar_scalar<mNumNodesPerCell>
            (tNumCells, mControlEntryOrdinal, aControls, aWorkSets.control());

        Plato::workset_config_scalar<mNumSpatialDims, mNumNodesPerCell>
            (tNumCells, mNodeCoordinate, aWorkSets.configuration());

        Plato::workset_state_scalar_scalar<mNumTimeStepsDofsPerNode, mNumNodesPerCell>
            (tNumCells, mScalarStateEntryOrdinal, aState.vector("time step"), aWorkSets.timeStep());

        if(aState.empty("artificial compressibility") == false)
        {
            Plato::workset_state_scalar_scalar<mNumACompressDofsPerNode, mNumNodesPerCell>
                (tNumCells, mScalarStateEntryOrdinal, aState.vector("artificial compressibility"), aWorkSets.artificialCompress());
        }
    }

    void setGradPrevTempWorkSets
    (const Plato::SpatialDomain & aDomain,
     const Plato::ScalarVector  & aControls,
     const PrimalStates         & aState,
           GradPrevTempWorkSets & aWorkSets)
    {
        Plato::workset_state_scalar_scalar<mNumVelDofsPerNode, mNumNodesPerCell>
            (aDomain, mVectorStateEntryOrdinal, aState.vector("current predictor"), aWorkSets.predictor());

        Plato::workset_state_scalar_scalar<mNumVelDofsPerNode, mNumNodesPerCell>
            (aDomain, mVectorStateEntryOrdinal, aState.vector("current velocity"), aWorkSets.currentVelocity());

        Plato::workset_state_scalar_scalar<mNumPressDofsPerNode, mNumNodesPerCell>
            (aDomain, mScalarStateEntryOrdinal, aState.vector("current pressure"), aWorkSets.currentPressure());

        Plato::workset_state_scalar_scalar<mNumTempDofsPerNode, mNumNodesPerCell>
            (aDomain, mScalarStateEntryOrdinal, aState.vector("current temperature"), aWorkSets.currentTemperature());

        Plato::workset_state_scalar_scalar<mNumVelDofsPerNode, mNumNodesPerCell>
            (aDomain, mVectorStateEntryOrdinal, aState.vector("previous velocity"), aWorkSets.previousVelocity());

        Plato::workset_state_scalar_scalar<mNumPressDofsPerNode, mNumNodesPerCell>
            (aDomain, mScalarStateEntryOrdinal, aState.vector("previous pressure"), aWorkSets.previousPressure());

        using PrevTempScalarT = typename GradPrevTempEvalT::PreviousEnergyScalarType;
        Plato::workset_state_scalar_fad<mNumTempDofsPerNode, mNumNodesPerCell, PrevTempScalarT>
            (aDomain, mScalarStateEntryOrdinal, aState.vector("previous temperature"), aWorkSets.previousTemperature());

        Plato::workset_control_scalar_scalar<mNumNodesPerCell>
            (aDomain, mControlEntryOrdinal, aControls, aWorkSets.control());

        Plato::workset_config_scalar<mNumSpatialDims, mNumNodesPerCell>
            (aDomain, mNodeCoordinate, aWorkSets.configuration());

        Plato::workset_state_scalar_scalar<mNumTimeStepsDofsPerNode, mNumNodesPerCell>
            (aDomain, mScalarStateEntryOrdinal, aState.vector("time step"), aWorkSets.timeStep());

        if(!aState.empty("artificial compressibility"))
        {
            Plato::workset_state_scalar_scalar<mNumACompressDofsPerNode, mNumNodesPerCell>
                (aDomain, mScalarStateEntryOrdinal, aState.vector("artificial compressibility"), aWorkSets.artificialCompress());
        }
    }

    void setGradPrevTempWorkSets
    (const Plato::ScalarVector  & aControls,
     const PrimalStates         & aState,
           GradPrevTempWorkSets & aWorkSets) const
    {
        auto tNumCells = aWorkSets.numCells();

        Plato::workset_state_scalar_scalar<mNumVelDofsPerNode, mNumNodesPerCell>
            (tNumCells, mVectorStateEntryOrdinal, aState.vector("current predictor"), aWorkSets.predictor());

        Plato::workset_state_scalar_scalar<mNumVelDofsPerNode, mNumNodesPerCell>
            (tNumCells, mVectorStateEntryOrdinal, aState.vector("current velocity"), aWorkSets.currentVelocity());

        Plato::workset_state_scalar_scalar<mNumPressDofsPerNode, mNumNodesPerCell>
            (tNumCells, mScalarStateEntryOrdinal, aState.vector("current pressure"), aWorkSets.currentPressure());

        Plato::workset_state_scalar_scalar<mNumTempDofsPerNode, mNumNodesPerCell>
            (tNumCells, mScalarStateEntryOrdinal, aState.vector("current temperature"), aWorkSets.currentTemperature());

        Plato::workset_state_scalar_scalar<mNumVelDofsPerNode, mNumNodesPerCell>
            (tNumCells, mVectorStateEntryOrdinal, aState.vector("previous velocity"), aWorkSets.previousVelocity());

        Plato::workset_state_scalar_scalar<mNumPressDofsPerNode, mNumNodesPerCell>
            (tNumCells, mScalarStateEntryOrdinal, aState.vector("previous pressure"), aWorkSets.previousPressure());

        using PrevTempScalarT = typename GradPrevTempEvalT::PreviousEnergyScalarType;
        Plato::workset_state_scalar_fad<mNumTempDofsPerNode, mNumNodesPerCell, PrevTempScalarT>
            (tNumCells, mScalarStateEntryOrdinal, aState.vector("previous temperature"), aWorkSets.previousTemperature());

        Plato::workset_control_scalar_scalar<mNumNodesPerCell>
            (tNumCells, mControlEntryOrdinal, aControls, aWorkSets.control());

        Plato::workset_config_scalar<mNumSpatialDims, mNumNodesPerCell>
            (tNumCells, mNodeCoordinate, aWorkSets.configuration());

        Plato::workset_state_scalar_scalar<mNumTimeStepsDofsPerNode, mNumNodesPerCell>
            (tNumCells, mScalarStateEntryOrdinal, aState.vector("time step"), aWorkSets.timeStep());

        if(aState.empty("artificial compressibility") == false)
        {
            Plato::workset_state_scalar_scalar<mNumACompressDofsPerNode, mNumNodesPerCell>
                (tNumCells, mScalarStateEntryOrdinal, aState.vector("artificial compressibility"), aWorkSets.artificialCompress());
        }
    }

    void setGradCurVelWorkSets
    (const Plato::SpatialDomain & aDomain,
     const Plato::ScalarVector  & aControls,
     const PrimalStates         & aState,
           GradCurVelWorkSets   & aWorkSets) const
    {
        Plato::workset_state_scalar_scalar<mNumVelDofsPerNode, mNumNodesPerCell>
            (aDomain, mVectorStateEntryOrdinal, aState.vector("current predictor"), aWorkSets.predictor());

        using CurVelScalarT = typename GradCurVelEvalT::CurrentMomentumScalarType;
        Plato::workset_state_scalar_fad<mNumVelDofsPerNode, mNumNodesPerCell, CurVelScalarT>
            (aDomain, mVectorStateEntryOrdinal, aState.vector("current velocity"), aWorkSets.currentVelocity());

        Plato::workset_state_scalar_scalar<mNumPressDofsPerNode, mNumNodesPerCell>
            (aDomain, mScalarStateEntryOrdinal, aState.vector("current pressure"), aWorkSets.currentPressure());

        Plato::workset_state_scalar_scalar<mNumTempDofsPerNode, mNumNodesPerCell>
            (aDomain, mScalarStateEntryOrdinal, aState.vector("current temperature"), aWorkSets.currentTemperature());

        Plato::workset_state_scalar_scalar<mNumVelDofsPerNode, mNumNodesPerCell>
            (aDomain, mVectorStateEntryOrdinal, aState.vector("previous velocity"), aWorkSets.previousVelocity());

        Plato::workset_state_scalar_scalar<mNumPressDofsPerNode, mNumNodesPerCell>
            (aDomain, mScalarStateEntryOrdinal, aState.vector("previous pressure"), aWorkSets.previousPressure());

        Plato::workset_state_scalar_scalar<mNumTempDofsPerNode, mNumNodesPerCell>
            (aDomain, mScalarStateEntryOrdinal, aState.vector("previous temperature"), aWorkSets.previousTemperature());

        Plato::workset_control_scalar_scalar<mNumNodesPerCell>
            (aDomain, mControlEntryOrdinal, aControls, aWorkSets.control());

        Plato::workset_config_scalar<mNumSpatialDims, mNumNodesPerCell>
            (aDomain, mNodeCoordinate, aWorkSets.configuration());

        Plato::workset_state_scalar_scalar<mNumTimeStepsDofsPerNode, mNumNodesPerCell>
            (aDomain, mScalarStateEntryOrdinal, aState.vector("time step"), aWorkSets.timeStep());

        if(!aState.empty("artificial compressibility"))
        {
            Plato::workset_state_scalar_scalar<mNumACompressDofsPerNode, mNumNodesPerCell>
                (aDomain, mScalarStateEntryOrdinal, aState.vector("artificial compressibility"), aWorkSets.artificialCompress());
        }
    }

    void setGradCurVelWorkSets
    (const Plato::ScalarVector & aControls,
     const PrimalStates        & aState,
           GradCurVelWorkSets  & aWorkSets) const
    {
        auto tNumCells = aWorkSets.numCells();

        Plato::workset_state_scalar_scalar<mNumVelDofsPerNode, mNumNodesPerCell>
            (tNumCells, mVectorStateEntryOrdinal, aState.vector("current predictor"), aWorkSets.predictor());

        using CurVelScalarT = typename GradCurVelEvalT::CurrentMomentumScalarType;
        Plato::workset_state_scalar_fad<mNumVelDofsPerNode, mNumNodesPerCell, CurVelScalarT>
            (tNumCells, mVectorStateEntryOrdinal, aState.vector("current velocity"), aWorkSets.currentVelocity());

        Plato::workset_state_scalar_scalar<mNumPressDofsPerNode, mNumNodesPerCell>
            (tNumCells, mScalarStateEntryOrdinal, aState.vector("current pressure"), aWorkSets.currentPressure());

        Plato::workset_state_scalar_scalar<mNumTempDofsPerNode, mNumNodesPerCell>
            (tNumCells, mScalarStateEntryOrdinal, aState.vector("current temperature"), aWorkSets.currentTemperature());

        Plato::workset_state_scalar_scalar<mNumVelDofsPerNode, mNumNodesPerCell>
            (tNumCells, mVectorStateEntryOrdinal, aState.vector("previous velocity"), aWorkSets.previousVelocity());

        Plato::workset_state_scalar_scalar<mNumPressDofsPerNode, mNumNodesPerCell>
            (tNumCells, mScalarStateEntryOrdinal, aState.vector("previous pressure"), aWorkSets.previousPressure());

        Plato::workset_state_scalar_scalar<mNumTempDofsPerNode, mNumNodesPerCell>
            (tNumCells, mScalarStateEntryOrdinal, aState.vector("previous temperature"), aWorkSets.previousTemperature());

        Plato::workset_control_scalar_scalar<mNumNodesPerCell>
            (tNumCells, mControlEntryOrdinal, aControls, aWorkSets.control());

        Plato::workset_config_scalar<mNumSpatialDims, mNumNodesPerCell>
            (tNumCells, mNodeCoordinate, aWorkSets.configuration());

        Plato::workset_state_scalar_scalar<mNumTimeStepsDofsPerNode, mNumNodesPerCell>
            (tNumCells, mScalarStateEntryOrdinal, aState.vector("time step"), aWorkSets.timeStep());

        if(aState.empty("artificial compressibility") == false)
        {
            Plato::workset_state_scalar_scalar<mNumACompressDofsPerNode, mNumNodesPerCell>
                (tNumCells, mScalarStateEntryOrdinal, aState.vector("artificial compressibility"), aWorkSets.artificialCompress());
        }
    }

    void setGradCurPressWorkSets
    (const Plato::SpatialDomain & aDomain,
     const Plato::ScalarVector  & aControls,
     const PrimalStates         & aState,
           GradCurPressWorkSets & aWorkSets) const
    {
        Plato::workset_state_scalar_scalar<mNumVelDofsPerNode, mNumNodesPerCell>
            (aDomain, mVectorStateEntryOrdinal, aState.vector("current predictor"), aWorkSets.predictor());

        Plato::workset_state_scalar_scalar<mNumVelDofsPerNode, mNumNodesPerCell>
            (aDomain, mVectorStateEntryOrdinal, aState.vector("current velocity"), aWorkSets.currentVelocity());

        using CurPressScalarT = typename GradCurPressEvalT::CurrentMassScalarType;
        Plato::workset_state_scalar_fad<mNumPressDofsPerNode, mNumNodesPerCell, CurPressScalarT>
            (aDomain, mScalarStateEntryOrdinal, aState.vector("current pressure"), aWorkSets.currentPressure());

        Plato::workset_state_scalar_scalar<mNumTempDofsPerNode, mNumNodesPerCell>
            (aDomain, mScalarStateEntryOrdinal, aState.vector("current temperature"), aWorkSets.currentTemperature());

        Plato::workset_state_scalar_scalar<mNumVelDofsPerNode, mNumNodesPerCell>
            (aDomain, mVectorStateEntryOrdinal, aState.vector("previous velocity"), aWorkSets.previousVelocity());

        Plato::workset_state_scalar_scalar<mNumPressDofsPerNode, mNumNodesPerCell>
            (aDomain, mScalarStateEntryOrdinal, aState.vector("previous pressure"), aWorkSets.previousPressure());

        Plato::workset_state_scalar_scalar<mNumTempDofsPerNode, mNumNodesPerCell>
            (aDomain, mScalarStateEntryOrdinal, aState.vector("previous temperature"), aWorkSets.previousTemperature());

        Plato::workset_control_scalar_scalar<mNumNodesPerCell>
            (aDomain, mControlEntryOrdinal, aControls, aWorkSets.control());

        Plato::workset_config_scalar<mNumSpatialDims, mNumNodesPerCell>
            (aDomain, mNodeCoordinate, aWorkSets.configuration());

        Plato::workset_state_scalar_scalar<mNumTimeStepsDofsPerNode, mNumNodesPerCell>
            (aDomain, mScalarStateEntryOrdinal, aState.vector("time step"), aWorkSets.timeStep());

        if(!aState.empty("artificial compressibility"))
        {
            Plato::workset_state_scalar_scalar<mNumACompressDofsPerNode, mNumNodesPerCell>
                (aDomain, mScalarStateEntryOrdinal, aState.vector("artificial compressibility"), aWorkSets.artificialCompress());
        }
    }

    void setGradCurPressWorkSets
    (const Plato::ScalarVector  & aControls,
     const PrimalStates         & aState,
           GradCurPressWorkSets & aWorkSets) const
    {
        auto tNumCells = aWorkSets.numCells();

        Plato::workset_state_scalar_scalar<mNumVelDofsPerNode, mNumNodesPerCell>
            (tNumCells, mVectorStateEntryOrdinal, aState.vector("current predictor"), aWorkSets.predictor());

        Plato::workset_state_scalar_scalar<mNumVelDofsPerNode, mNumNodesPerCell>
            (tNumCells, mVectorStateEntryOrdinal, aState.vector("current velocity"), aWorkSets.currentVelocity());

        using CurPressScalarT = typename GradCurPressEvalT::CurrentMassScalarType;
        Plato::workset_state_scalar_fad<mNumPressDofsPerNode, mNumNodesPerCell, CurPressScalarT>
            (tNumCells, mScalarStateEntryOrdinal, aState.vector("current pressure"), aWorkSets.currentPressure());

        Plato::workset_state_scalar_scalar<mNumTempDofsPerNode, mNumNodesPerCell>
            (tNumCells, mScalarStateEntryOrdinal, aState.vector("current temperature"), aWorkSets.currentTemperature());

        Plato::workset_state_scalar_scalar<mNumVelDofsPerNode, mNumNodesPerCell>
            (tNumCells, mVectorStateEntryOrdinal, aState.vector("previous velocity"), aWorkSets.previousVelocity());

        Plato::workset_state_scalar_scalar<mNumPressDofsPerNode, mNumNodesPerCell>
            (tNumCells, mScalarStateEntryOrdinal, aState.vector("previous pressure"), aWorkSets.previousPressure());

        Plato::workset_state_scalar_scalar<mNumTempDofsPerNode, mNumNodesPerCell>
            (tNumCells, mScalarStateEntryOrdinal, aState.vector("previous temperature"), aWorkSets.previousTemperature());

        Plato::workset_control_scalar_scalar<mNumNodesPerCell>
            (tNumCells, mControlEntryOrdinal, aControls, aWorkSets.control());

        Plato::workset_config_scalar<mNumSpatialDims, mNumNodesPerCell>
            (tNumCells, mNodeCoordinate, aWorkSets.configuration());

        Plato::workset_state_scalar_scalar<mNumTimeStepsDofsPerNode, mNumNodesPerCell>
            (tNumCells, mScalarStateEntryOrdinal, aState.vector("time step"), aWorkSets.timeStep());

        if(aState.empty("artificial compressibility") == false)
        {
            Plato::workset_state_scalar_scalar<mNumACompressDofsPerNode, mNumNodesPerCell>
                (tNumCells, mScalarStateEntryOrdinal, aState.vector("artificial compressibility"), aWorkSets.artificialCompress());
        }
    }

    void setGradCurTempWorkSets
    (const Plato::SpatialDomain & aDomain,
     const Plato::ScalarVector  & aControls,
     const PrimalStates         & aState,
           GradCurTempWorkSets  & aWorkSets) const
    {
        Plato::workset_state_scalar_scalar<mNumVelDofsPerNode, mNumNodesPerCell>
            (aDomain, mVectorStateEntryOrdinal, aState.vector("current predictor"), aWorkSets.predictor());

        Plato::workset_state_scalar_scalar<mNumVelDofsPerNode, mNumNodesPerCell>
            (aDomain, mVectorStateEntryOrdinal, aState.vector("current velocity"), aWorkSets.currentVelocity());

        Plato::workset_state_scalar_scalar<mNumPressDofsPerNode, mNumNodesPerCell>
            (aDomain, mScalarStateEntryOrdinal, aState.vector("current pressure"), aWorkSets.currentPressure());

        using CurTempScalarT = typename GradCurTempEvalT::CurrentEnergyScalarType;
        Plato::workset_state_scalar_fad<mNumTempDofsPerNode, mNumNodesPerCell, CurTempScalarT>
            (aDomain, mScalarStateEntryOrdinal, aState.vector("current temperature"), aWorkSets.currentTemperature());

        Plato::workset_state_scalar_scalar<mNumVelDofsPerNode, mNumNodesPerCell>
            (aDomain, mVectorStateEntryOrdinal, aState.vector("previous velocity"), aWorkSets.previousVelocity());

        Plato::workset_state_scalar_scalar<mNumPressDofsPerNode, mNumNodesPerCell>
            (aDomain, mScalarStateEntryOrdinal, aState.vector("previous pressure"), aWorkSets.previousPressure());

        Plato::workset_state_scalar_scalar<mNumTempDofsPerNode, mNumNodesPerCell>
            (aDomain, mScalarStateEntryOrdinal, aState.vector("previous temperature"), aWorkSets.previousTemperature());

        Plato::workset_control_scalar_scalar<mNumNodesPerCell>
            (aDomain, mControlEntryOrdinal, aControls, aWorkSets.control());

        Plato::workset_config_scalar<mNumSpatialDims, mNumNodesPerCell>
            (aDomain, mNodeCoordinate, aWorkSets.configuration());

        Plato::workset_state_scalar_scalar<mNumTimeStepsDofsPerNode, mNumNodesPerCell>
            (aDomain, mScalarStateEntryOrdinal, aState.vector("time step"), aWorkSets.timeStep());

        if(!aState.empty("artificial compressibility"))
        {
            Plato::workset_state_scalar_scalar<mNumACompressDofsPerNode, mNumNodesPerCell>
                (aDomain, mScalarStateEntryOrdinal, aState.vector("artificial compressibility"), aWorkSets.artificialCompress());
        }
    }

    void setGradCurTempWorkSets
    (const Plato::ScalarVector & aControls,
     const PrimalStates        & aState,
           GradCurTempWorkSets & aWorkSets) const
    {
        auto tNumCells = aWorkSets.numCells();

        Plato::workset_state_scalar_scalar<mNumVelDofsPerNode, mNumNodesPerCell>
            (tNumCells, mVectorStateEntryOrdinal, aState.vector("current predictor"), aWorkSets.predictor());

        Plato::workset_state_scalar_scalar<mNumVelDofsPerNode, mNumNodesPerCell>
            (tNumCells, mVectorStateEntryOrdinal, aState.vector("current velocity"), aWorkSets.currentVelocity());

        Plato::workset_state_scalar_scalar<mNumPressDofsPerNode, mNumNodesPerCell>
            (tNumCells, mScalarStateEntryOrdinal, aState.vector("current pressure"), aWorkSets.currentPressure());

        using CurTempScalarT = typename GradCurTempEvalT::CurrentEnergyScalarType;
        Plato::workset_state_scalar_scalar<mNumTempDofsPerNode, mNumNodesPerCell, CurTempScalarT>
            (tNumCells, mScalarStateEntryOrdinal, aState.vector("current temperature"), aWorkSets.currentTemperature());

        Plato::workset_state_scalar_scalar<mNumVelDofsPerNode, mNumNodesPerCell>
            (tNumCells, mVectorStateEntryOrdinal, aState.vector("previous velocity"), aWorkSets.previousVelocity());

        Plato::workset_state_scalar_scalar<mNumPressDofsPerNode, mNumNodesPerCell>
            (tNumCells, mScalarStateEntryOrdinal, aState.vector("previous pressure"), aWorkSets.previousPressure());

        Plato::workset_state_scalar_scalar<mNumTempDofsPerNode, mNumNodesPerCell>
            (tNumCells, mScalarStateEntryOrdinal, aState.vector("previous temperature"), aWorkSets.previousTemperature());

        Plato::workset_control_scalar_scalar<mNumNodesPerCell>
            (tNumCells, mControlEntryOrdinal, aControls, aWorkSets.control());

        Plato::workset_config_scalar<mNumSpatialDims, mNumNodesPerCell>
            (tNumCells, mNodeCoordinate, aWorkSets.configuration());

        Plato::workset_state_scalar_scalar<mNumTimeStepsDofsPerNode, mNumNodesPerCell>
            (tNumCells, mScalarStateEntryOrdinal, aState.vector("time step"), aWorkSets.timeStep());

        if(aState.empty("artificial compressibility") == false)
        {
            Plato::workset_state_scalar_scalar<mNumACompressDofsPerNode, mNumNodesPerCell>
                (tNumCells, mScalarStateEntryOrdinal, aState.vector("artificial compressibility"), aWorkSets.artificialCompress());
        }
    }
};
// class VectorFunction

template<typename PhysicsT>
class CriterionFactory
{
private:
    using ScalarFunctionType = std::shared_ptr<Plato::FluidMechanics::CriterionBase>;

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
     * \brief Creates criterion interface, which allows evaluations.
     * \param [in] aSpatialModel  C++ structure with volume and surface mesh databases
     * \param [in] aDataMap       Plato Analyze data map
     * \param [in] aInputs        input parameters from Analyze's input file
     * \param [in] aName          scalar function name
     **********************************************************************************/
    ScalarFunctionType
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
                std::make_shared<Plato::FluidMechanics::PhysicsScalarFunction<PhysicsT>>
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
    std::shared_ptr<Plato::FluidMechanics::AbstractVectorFunction<PhysicsT, EvaluationT>>
    createVectorFunction
    (const std::string          & aTag,
     const Plato::SpatialDomain & aDomain,
     Plato::DataMap             & aDataMap,
     Teuchos::ParameterList     & aInputs)
    {
        if( !aInputs.isSublist(aTag) == false )
        {
            THROWERR(std::string("Vector function with tag '") + aTag + "' is not supported.")
        }

        auto tFunParams = aInputs.sublist(aTag);
        auto tLowerTag = Plato::tolower(aTag);
        // TODO: Add pressure, velocity, temperature, and predictor element residuals. explore function interface
        if( tLowerTag == "pressure" )
        {
            return ( std::make_shared<Plato::FluidMechanics::PressureIncrementResidual<PhysicsT, EvaluationT>>(aDomain, aDataMap, aInputs) );
        }
        else if( tLowerTag == "velocity" )
        {
            return ( std::make_shared<Plato::FluidMechanics::VelocityIncrementResidual<PhysicsT, EvaluationT>>(aDomain, aDataMap, aInputs) );
        }
        else if( tLowerTag == "temperature" )
        {
            return ( std::make_shared<Plato::FluidMechanics::TemperatureIncrementResidual<PhysicsT, EvaluationT>>(aDomain, aDataMap, aInputs) );
        }
        else if( tLowerTag == "velocity predictor" )
        {
            return ( std::make_shared<Plato::FluidMechanics::VelocityPredictorResidual<PhysicsT, EvaluationT>>(aDomain, aDataMap, aInputs) );
        }
        else
        {
            THROWERR(std::string("Vector function with tag '") + aTag + "' is not supported.")
        }
    }

    template <typename PhysicsT, typename EvaluationT>
    std::shared_ptr<Plato::FluidMechanics::AbstractScalarFunction<PhysicsT, EvaluationT>>
    createScalarFunction
    (const std::string          & aType,
     const std::string          & aTag,
     const Plato::SpatialDomain & aDomain,
     Plato::DataMap             & aDataMap,
     Teuchos::ParameterList     & aInputs)
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
        if( tLowerTag == "average surface pressure" )
        {
            return ( std::make_shared<Plato::FluidMechanics::AverageSurfacePressure<PhysicsT, EvaluationT>>
                (tLowerTag, aDomain, aDataMap, aInputs) );
        }
        else
        {
            THROWERR(std::string("Scalar function of type '") + aType + "' with tag ' " + aTag + "' is not supported.")
        }
    }
};
// struct FunctionFactory





// todo: finish weighted scalar function
template<typename PhysicsT>
class WeightedScalarFunction : public Plato::FluidMechanics::CriterionBase
{
private:
    // static metadata
    static constexpr auto mNumSpatialDims         = PhysicsT::SimplexT::mNumSpatialDims;         /*!< number of spatial dimensions */
    static constexpr auto mNumPressDofsPerNode    = PhysicsT::SimplexT::mNumMassDofsPerNode;     /*!< number of mass dofs per node */
    static constexpr auto mNumTempDofsPerNode     = PhysicsT::SimplexT::mNumEnergyDofsPerNode;   /*!< number of energy dofs per node */
    static constexpr auto mNumVelDofsPerNode      = PhysicsT::SimplexT::mNumMomentumDofsPerNode; /*!< number of momentum dofs per node */
    static constexpr auto mNumControlsPerNode     = PhysicsT::SimplexT::mNumControl;             /*!< number of design variables per node */

    // set local typenames
    using PrimalStates = Plato::FluidMechanics::States;
    using Criterion    = std::shared_ptr<Plato::FluidMechanics::CriterionBase>;

    bool mDiagnostics; /*!< write diagnostics to terminal */
    Plato::DataMap& mDataMap; /*!< output database */
    const Plato::SpatialModel& mSpatialModel; /*!< mesh database */

    std::string mFuncName; /*!< weighted scalar function name */

    std::vector<Criterion>     mCriteria;         /*!< list of scalar function criteria */
    std::vector<std::string>   mCriterionNames;   /*!< list of criterion names */
    std::vector<Plato::Scalar> mCriterionWeights; /*!< list of criterion weights */

public:
    WeightedScalarFunction
    (const Plato::SpatialModel & aSpatialModel,
     Plato::DataMap & aDataMap,
     Teuchos::ParameterList & aInputs,
     std::string & aName) :
         mDiagnostics(false),
         mDataMap     (aDataMap),
         mSpatialModel(aSpatialModel),
         mFuncName    (aName)
    {
        this->initialize(aInputs);
    }

    virtual ~WeightedScalarFunction(){}

    void append
    (const Criterion     & aFunc,
     const std::string   & aName,
           Plato::Scalar   aWeight = 1.0)
    {
        mCriteria.push_back(aFunc);
        mCriterionNames.push_back(aName);
        mCriterionWeights.push_back(aWeight);
    }

    std::string name() const override
    {
        return mFuncName;
    }

    Plato::Scalar
    value
    (const Plato::ScalarVector & aControls,
     const PrimalStates & aStates) const override
    {
        Plato::Scalar tResult = 0.0;
        for (auto& tCriterion : mCriteria)
        {
            auto tIndex = &tCriterion - &mCriteria[0];
            const auto tValue = tCriterion->value(aControls, aStates);
            const auto tFuncWeight = mCriterionWeights[tIndex];
            const auto tFuncValue = tFuncWeight * tValue;

            const auto tFuncName = mCriterionNames[tIndex];
            mDataMap.mScalarValues[tFuncName] = tFuncValue;
            tResult += tFuncValue;

            if(mDiagnostics)
            {
                printf("Scalar Function Name = %s \t Value = %f\n", tFuncName.c_str(), tFuncValue);
            }
        }

        if(mDiagnostics)
        {
            printf("Weighted Sum Name = %s \t Value = %f\n", mFuncName.c_str(), tResult);
        }
        return tResult;
    }

    Plato::ScalarVector gradientConfig
    (const Plato::ScalarVector & aControls,
     const PrimalStates & aStates) const override
    {
        const auto tNumNodes = mSpatialModel.Mesh.nverts();
        Plato::ScalarVector tTotalDerivative("total derivative", mNumSpatialDims * tNumNodes);
        for (auto& tCriterion : mCriteria)
        {
            auto tIndex = &tCriterion - &mCriteria[0];
            auto tGradient = tCriterion->gradientConfig(aControls, aStates);
            const auto tFuncWeight = mCriterionWeights[tIndex];
            Plato::blas1::update(tFuncWeight, tGradient, static_cast<Plato::Scalar>(1.0), tTotalDerivative);
        }
        return tTotalDerivative;
    }

    Plato::ScalarVector gradientControl
    (const Plato::ScalarVector & aControls,
     const PrimalStates & aStates) const override
    {
        const auto tNumNodes = mSpatialModel.Mesh.nverts();
        Plato::ScalarVector tTotalDerivative("total derivative", mNumControlsPerNode * tNumNodes);
        for (auto& tCriterion : mCriteria)
        {
            auto tIndex = &tCriterion - &mCriteria[0];
            auto tGradient = tCriterion->gradientControl(aControls, aStates);
            const auto tFuncWeight = mCriterionWeights[tIndex];
            Plato::blas1::update(tFuncWeight, tGradient, static_cast<Plato::Scalar>(1.0), tTotalDerivative);
        }
        return tTotalDerivative;
    }

    Plato::ScalarVector gradientCurrentPress
    (const Plato::ScalarVector & aControls,
     const PrimalStates & aStates) const override
    {
        const auto tNumNodes = mSpatialModel.Mesh.nverts();
        Plato::ScalarVector tTotalDerivative("total derivative", mNumPressDofsPerNode * tNumNodes);
        for (auto& tCriterion : mCriteria)
        {
            auto tIndex = &tCriterion - &mCriteria[0];
            auto tGradient = tCriterion->gradientCurrentPress(aControls, aStates);
            const auto tFuncWeight = mCriterionWeights[tIndex];
            Plato::blas1::update(tFuncWeight, tGradient, static_cast<Plato::Scalar>(1.0), tTotalDerivative);
        }
        return tTotalDerivative;
    }

    Plato::ScalarVector gradientCurrentTemp
    (const Plato::ScalarVector & aControls,
     const PrimalStates & aStates) const override
    {
        const auto tNumNodes = mSpatialModel.Mesh.nverts();
        Plato::ScalarVector tTotalDerivative("total derivative", mNumTempDofsPerNode * tNumNodes);
        for (auto& tCriterion : mCriteria)
        {
            auto tIndex = &tCriterion - &mCriteria[0];
            auto tGradient = tCriterion->gradientCurrentTemp(aControls, aStates);
            const auto tFuncWeight = mCriterionWeights[tIndex];
            Plato::blas1::update(tFuncWeight, tGradient, static_cast<Plato::Scalar>(1.0), tTotalDerivative);
        }
        return tTotalDerivative;
    }

    Plato::ScalarVector gradientCurrentVel
    (const Plato::ScalarVector & aControls,
     const PrimalStates & aStates) const override
    {
        const auto tNumNodes = mSpatialModel.Mesh.nverts();
        Plato::ScalarVector tTotalDerivative("total derivative", mNumVelDofsPerNode * tNumNodes);
        for (auto& tCriterion : mCriteria)
        {
            auto tIndex = &tCriterion - &mCriteria[0];
            auto tGradient = tCriterion->gradientCurrentVel(aControls, aStates);
            const auto tFuncWeight = mCriterionWeights[tIndex];
            Plato::blas1::update(tFuncWeight, tGradient, static_cast<Plato::Scalar>(1.0), tTotalDerivative);
        }
        return tTotalDerivative;
    }

private:
    void initialize(Teuchos::ParameterList & aInputs)
    {
        if(aInputs.sublist("Criteria").isSublist(mFuncName) == false)
        {
            THROWERR(std::string("Scalar function with tag '") + mFuncName + "' is not defined in the input file.")
        }
        auto tCriteriaInputs = aInputs.sublist("Criteria").sublist(mFuncName);

        mCriterionNames   = Plato::parse_criterion_names(tCriteriaInputs);
        mCriterionWeights = Plato::parse_criterion_weights(tCriteriaInputs);
        if (mCriterionNames.size() != mCriterionWeights.size())
        {
            THROWERR(std::string("Dimensions mismatch.  Number of 'Functions' and 'Weights' do not match. ") +
                     "Check scalar function with name '" + mFuncName + "'.")
        }

        Plato::FluidMechanics::CriterionFactory<PhysicsT> tFactory;
        for(auto& tName : mCriterionNames)
        {
            auto tScalarFunction = tFactory.createCriterion(mSpatialModel, mDataMap, aInputs, tName);
            mCriteria.push_back(tScalarFunction);
        }
    }
};
// class WeightedScalarFunction

}
// namespace FluidMechanics


// todo: physics types
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

    typedef Plato::FluidMechanics::FunctionFactory FunctionFactory;
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
(const Plato::FluidMechanics::States & aStates,
 Plato::Scalar aCritialCompresibility = 0.5)
{
    auto tPrandtl = aStates.scalar("prandtl");
    auto tReynolds = aStates.scalar("reynolds");
    auto tElemSize = aStates.vector("element size");
    auto tVelocity = aStates.vector("current velocity");

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
            && (tConvectiveVelocity < aCritialCompresibility) ? tConvectiveVelocity : aCritialCompresibility;
        tArtificialCompressibility = (tDiffusionVelocity < tConvectiveVelocity ) && (tDiffusionVelocity < tThermalVelocity)
            && (tDiffusionVelocity < aCritialCompresibility) ? tDiffusionVelocity : tArtificialCompressibility;
        tArtificialCompressibility = (tThermalVelocity < tConvectiveVelocity ) && (tThermalVelocity < tDiffusionVelocity)
            && (tThermalVelocity < aCritialCompresibility) ? tThermalVelocity : tArtificialCompressibility;

        tArtificalCompress(aOrdinal) = tArtificialCompressibility;
    }, "calculate artificial compressibility");

    return tArtificalCompress;
}

inline Plato::ScalarVector
calculate_stable_time_step
(const Plato::FluidMechanics::States & aStates)
{
    auto tElemSize = aStates.vector("element size");
    auto tVelocity = aStates.vector("current velocity");
    auto tArtificialCompressibility = aStates.vector("artificial compressibility");

    auto tReynolds = aStates.scalar("reynolds");
    auto tSafetyFactor = aStates.scalar("time step safety factor");

    auto tLength = tVelocity.size();
    Plato::ScalarVector tTimeStep("time step", tLength);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tLength), LAMBDA_EXPRESSION(const Plato::OrdinalType & aOrdinal)
    {
        // calculate convective velocity
        Plato::Scalar tConvectiveVelocity = tVelocity(aOrdinal) * tVelocity(aOrdinal);
        tConvectiveVelocity = sqrt(tConvectiveVelocity);
        auto tCriticalConvectiveTimeStep = tElemSize(aOrdinal) /
            (tConvectiveVelocity + tArtificialCompressibility(aOrdinal));

        // calculate diffusive velocity
        auto tDiffusiveVelocity = static_cast<Plato::Scalar>(2.0) / (tElemSize(aOrdinal) * tReynolds);
        auto tCriticalDiffusiveTimeStep = tElemSize(aOrdinal) / tDiffusiveVelocity;

        // calculate stable time step
        auto tCriticalTimeStep = tCriticalConvectiveTimeStep < tCriticalDiffusiveTimeStep ?
            tCriticalConvectiveTimeStep : tCriticalDiffusiveTimeStep;
        tTimeStep(aOrdinal) = tCriticalTimeStep * tSafetyFactor;
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

inline Plato::ScalarVector
calculate_pressure_residual
(const Plato::ScalarVector& aTimeStep,
 const Plato::ScalarVector& aCurrentState,
 const Plato::ScalarVector& aPreviousState,
 const Plato::ScalarVector& aArtificialCompressibility)
{
    // calculate stopping criterion, which is defined as
    // \frac{1}{\beta^2} \left( \frac{p^{n} - p^{n-1}}{\Delta{t}}\right ),
    // where \beta denotes the artificial compressibility
    auto tLength = aCurrentState.size();
    Plato::ScalarVector tResidual("pressure residual", tLength);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tLength), LAMBDA_EXPRESSION(const Plato::OrdinalType & aOrdinal)
    {
        auto tDeltaPressOverTimeStep = ( aCurrentState(aOrdinal) - aPreviousState(aOrdinal) ) / aTimeStep(aOrdinal);
        auto tOneOverBetaSquared = static_cast<Plato::Scalar>(1) /
            ( aArtificialCompressibility(aOrdinal) * aArtificialCompressibility(aOrdinal) );
        tResidual(aOrdinal) = tOneOverBetaSquared * tDeltaPressOverTimeStep;
    }, "calculate stopping criterion");

    return tResidual;
}

inline Plato::Scalar
calculate_explicit_solve_convergence_criterion
(const Plato::FluidMechanics::States & aStates)
{
    auto tTimeStep = aStates.vector("time step");
    auto tCurrentPressure = aStates.vector("current pressure");
    auto tPreviousPressure = aStates.vector("previous pressure");
    auto tArtificialCompress = aStates.vector("artificial compressibility");
    auto tResidual = Plato::cbs::calculate_pressure_residual(tTimeStep, tCurrentPressure, tPreviousPressure, tArtificialCompress);
    auto tStoppingCriterion = Plato::blas1::dot(tResidual, tResidual);
    return tStoppingCriterion;
}

inline Plato::Scalar
calculate_semi_implicit_solve_convergence_criterion
(const Plato::FluidMechanics::States & aStates)
{
    std::vector<Plato::Scalar> tErrors;

    // pressure error
    auto tTimeStep = aStates.vector("time step");
    auto tCurrentState = aStates.vector("current pressure");
    auto tPreviousState = aStates.vector("previous pressure");
    auto tArtificialCompress = aStates.vector("artificial compressibility");
    auto tMyResidual = Plato::cbs::calculate_pressure_residual(tTimeStep, tCurrentState, tPreviousState, tArtificialCompress);
    Plato::Scalar tInfinityNorm = 0.0;
    Plato::blas1::max(tMyResidual, tInfinityNorm);
    return tInfinityNorm;
}

}
// namespace cbs

namespace FluidMechanics
{

class AbstractProblem
{
public:
    virtual ~AbstractProblem() {}

    virtual void output(std::string aFilePath) = 0;
    virtual const Plato::DataMap& getDataMap() const = 0;
    virtual Plato::Solutions solution(const Plato::ScalarVector& aControl) = 0;
    virtual Plato::Scalar criterionValue(const Plato::ScalarVector & aControl, const std::string& aName) = 0;
    virtual Plato::ScalarVector criterionGradient(const Plato::ScalarVector &aControl, const std::string &aName) = 0;
    virtual Plato::ScalarVector criterionGradientX(const Plato::ScalarVector &aControl, const std::string &aName) = 0;
};

template<typename PhysicsT>
class FluidMechanicsProblem : public Plato::FluidMechanics::AbstractProblem
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
    Plato::Scalar mTimeStepSafetyFactor = 0.5; /*!< safety factor applied to stable time step */
    Plato::OrdinalType mNumTimeSteps = 100;

    Plato::ScalarMultiVector mPressure;
    Plato::ScalarMultiVector mVelocity;
    Plato::ScalarMultiVector mPredictor;
    Plato::ScalarMultiVector mTemperature;

    Plato::FluidMechanics::VectorFunction<typename PhysicsT::MassPhysicsT>     mPressureResidual;
    Plato::FluidMechanics::VectorFunction<typename PhysicsT::MomentumPhysicsT> mPredictorResidual;
    Plato::FluidMechanics::VectorFunction<typename PhysicsT::MomentumPhysicsT> mVelocityResidual;
    Plato::FluidMechanics::VectorFunction<typename PhysicsT::EnergyPhysicsT>   mTemperatureResidual;

    using Criterion = std::shared_ptr<Plato::FluidMechanics::CriterionBase>;
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

    using DualStates = Plato::FluidMechanics::Dual;
    using PrimalStates = Plato::FluidMechanics::States;

public:
    FluidMechanicsProblem
    (Omega_h::Mesh          & aMesh,
     Omega_h::MeshSets      & aMeshSets,
     Teuchos::ParameterList & aInputs,
     Comm::Machine          & aMachine) :
         mSpatialModel       (aMesh, aMeshSets, aInputs),
         mPressureResidual   ("Pressure", mSpatialModel, mDataMap, aInputs),
         mVelocityResidual   ("Velocity", mSpatialModel, mDataMap, aInputs),
         mPredictorResidual  ("Velocity Predictor", mSpatialModel, mDataMap, aInputs),
         mTemperatureResidual("Temperature", mSpatialModel, mDataMap, aInputs)
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
        for(Plato::OrdinalType tStep = 0; tStep < tTimeSteps; tStep++)
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

    Plato::Solutions solution
    (const Plato::ScalarVector& aControl)
    {
        PrimalStates tStates;
        this->calculateElemCharacteristicSize(tStates);

        for (Plato::OrdinalType tStep = 1; tStep < mNumTimeSteps; tStep++)
        {
            tStates.scalar("step", tStep);
            this->setPrimalStates(tStates);
            this->calculateStableTimeSteps(tStates);

            this->updatePredictor(aControl, tStates);
            this->updatePressure(aControl, tStates);
            this->updateVelocity(aControl, tStates);
            this->updateTemperature(aControl, tStates);

            // todo: verify BC enforcement
            this->enforceVelocityBoundaryConditions(tStates);
            this->enforcePressureBoundaryConditions(tStates);
            this->enforceTemperatureBoundaryConditions(tStates);

            if(this->checkStoppingCriteria(tStates))
            {
                break;
            }
        }

        Plato::Solutions tSolution;
        tSolution.set("mass state", mPressure);
        tSolution.set("energy state", mTemperature);
        tSolution.set("momentum state", mVelocity);
        return tSolution;
    }

    Plato::Scalar criterionValue
    (const Plato::ScalarVector & aControl,
     const std::string         & aName)
    {
        auto tItr = mCriteria.find(aName);
        if (tItr == mCriteria.end())
        {
            PrimalStates tPrimalStates;
            this->calculateElemCharacteristicSize(tPrimalStates);

            Plato::Scalar tOutput(0);
            auto tNumTimeSteps = mVelocity.extent(0);
            for (Plato::OrdinalType tStep = 0; tStep < tNumTimeSteps; tStep++)
            {
                tPrimalStates.scalar("step", tStep);
                auto tPressure = Kokkos::subview(mPressure, tStep, Kokkos::ALL());
                auto tVelocity = Kokkos::subview(mVelocity, tStep, Kokkos::ALL());
                auto tTemperature = Kokkos::subview(mTemperature, tStep, Kokkos::ALL());
                tPrimalStates.vector("current pressure", tPressure);
                tPrimalStates.vector("current velocity", tVelocity);
                tPrimalStates.vector("current temperature", tTemperature);
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
     const std::string         & aName)
    {
        auto tItr = mCriteria.find(aName);
        if (tItr == mCriteria.end())
        {
            DualStates   tDualStates;
            PrimalStates tCurrentStates, tPreviousStates;
            this->calculateElemCharacteristicSize(tCurrentStates);

            auto tLastStepIndex = mVelocity.extent(0) - 1;
            Plato::ScalarVector tTotalDerivative("Total Derivative", aControl.size());
            for(Plato::OrdinalType tStep = tLastStepIndex; tStep >= 0; tStep--)
            {
                tDualStates.scalar("step", tStep);
                tCurrentStates.scalar("step", tStep);
                tPreviousStates.scalar("step", tStep + 1);

                this->setDualStates(tDualStates);
                this->setPrimalStates(tCurrentStates);
                this->setPrimalStates(tPreviousStates);

                this->calculateStableTimeSteps(tCurrentStates);
                this->calculateStableTimeSteps(tPreviousStates);

                this->calculateVelocityAdjoint(aName, aControl, tCurrentStates, tPreviousStates, tDualStates);
                this->calculateTemperatureAdjoint(aName, aControl, tCurrentStates, tPreviousStates, tDualStates);
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
     const std::string         & aName)
    {
        auto tItr = mCriteria.find(aName);
        if (tItr == mCriteria.end())
        {
            DualStates   tDualStates;
            PrimalStates tCurrentStates, tPreviousStates;
            this->calculateElemCharacteristicSize(tCurrentStates);

            auto tLastStepIndex = mVelocity.extent(0) - 1;
            Plato::ScalarVector tTotalDerivative("Total Derivative", aControl.size());
            for(Plato::OrdinalType tStep = tLastStepIndex; tStep >= 0; tStep--)
            {
                tDualStates.scalar("step", tStep);
                tCurrentStates.scalar("step", tStep);
                tPreviousStates.scalar("step", tStep + 1);

                this->setDualStates(tDualStates);
                this->setPrimalStates(tCurrentStates);
                this->setPrimalStates(tPreviousStates);

                this->calculateStableTimeSteps(tCurrentStates);
                this->calculateStableTimeSteps(tPreviousStates);

                this->calculateVelocityAdjoint(aName, aControl, tCurrentStates, tPreviousStates, tDualStates);
                this->calculateTemperatureAdjoint(aName, aControl, tCurrentStates, tPreviousStates, tDualStates);
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
    (Teuchos::ParameterList & aInputs,
     Comm::Machine          & aMachine)
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
        mPredictor   = Plato::ScalarMultiVector("Predictor Snapshots", mNumTimeSteps, tNumNodes * mNumVelDofsPerNode);
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
            Plato::FluidMechanics::CriterionFactory<PhysicsT> tScalarFuncFactory;

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

            Plato::OrdinalType tStep = aStates.scalar("step");
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
        aStates.vector("element size", tElemCharacteristicSize);
    }

    void calculateStableTimeSteps(PrimalStates & aStates)
    {
        auto tArtificialCompressibility = Plato::cbs::calculate_artificial_compressibility(aStates);
        aStates.vector("artificial compressibility", tArtificialCompressibility);

        aStates.scalar("time step safety factor", mTimeStepSafetyFactor);
        auto tTimeStep = Plato::cbs::calculate_stable_time_step(aStates);
        if(mIsTransientProblem)
        {
            Plato::Scalar tMinTimeStep(0);
            Plato::blas1::min(tTimeStep, tMinTimeStep);
            Plato::blas1::fill(tMinTimeStep, tTimeStep);
            auto tCurrentTimeStepIndex = aStates.scalar("step index");
            auto tCurrentTime = tMinTimeStep * static_cast<Plato::Scalar>(tCurrentTimeStepIndex);
            aStates.scalar("current time", tCurrentTime);
        }
        aStates.vector("time step", tTimeStep);
    }

    void enforceVelocityBoundaryConditions(PrimalStates & aStates)
    {
        Plato::ScalarVector tBcValues;
        Plato::LocalOrdinalVector tBcDofs;
        if(mIsTransientProblem)
        {
            auto tCurrentTime = aStates.scalar("current time");
            mVelocityStateBoundaryConditions.get(tBcDofs, tBcValues, tCurrentTime);
        }
        else
        {
            mVelocityStateBoundaryConditions.get(tBcDofs, tBcValues);
        }
        auto tCurrentVelocity = aStates.vector("current velocity");
        Plato::cbs::enforce_boundary_condition(tBcDofs, tBcValues, tCurrentVelocity);
    }

    void enforcePressureBoundaryConditions(PrimalStates& aStates)
    {
        Plato::ScalarVector tBcValues;
        Plato::LocalOrdinalVector tBcDofs;
        if(mIsTransientProblem)
        {
            auto tCurrentTime = aStates.scalar("current time");
            mPressureStateBoundaryConditions.get(tBcDofs, tBcValues, tCurrentTime);
        }
        else
        {
            mPressureStateBoundaryConditions.get(tBcDofs, tBcValues);
        }
        auto tCurrentPressure = aStates.vector("current pressure");
        Plato::cbs::enforce_boundary_condition(tBcDofs, tBcValues, tCurrentPressure);
    }

    void enforceTemperatureBoundaryConditions(PrimalStates & aStates)
    {
        Plato::ScalarVector tBcValues;
        Plato::LocalOrdinalVector tBcDofs;
        if(mIsTransientProblem)
        {
            auto tCurrentTime = aStates.scalar("current time");
            mTemperatureStateBoundaryConditions.get(tBcDofs, tBcValues, tCurrentTime);
        }
        else
        {
            mTemperatureStateBoundaryConditions.get(tBcDofs, tBcValues);
        }
        auto tCurrentTemperature = aStates.vector("current temperature");
        Plato::cbs::enforce_boundary_condition(tBcDofs, tBcValues, tCurrentTemperature);
    }

    void setDualStates(DualStates & aStates)
    {
        if(aStates.isVectorMapEmpty())
        {
            // FIRST BACKWARD TIME INTEGRATION STEP
            auto tTotalNumNodes = mSpatialModel.Mesh.nverts();
            std::vector<std::string> tNames =
                {"current pressure adjoint" , "current temperature adjoint",
                "previous pressure adjoint", "previous temperature adjoint"};
            for(auto& tName : tNames)
            {
                Plato::ScalarVector tView(tName, tTotalNumNodes);
                aStates.vector(tName, tView);
            }

            auto tTotalNumDofs = mNumVelDofsPerNode * tTotalNumNodes;
            tNames = {"current velocity adjoint" , "current predictor adjoint" ,
                      "previous velocity adjoint", "previous predictor adjoint"};
            for(auto& tName : tNames)
            {
                Plato::ScalarVector tView(tName, tTotalNumDofs);
                aStates.vector(tName, tView);
            }
        }
        else
        {
            // N-TH BACKWARD TIME INTEGRATION STEP
            std::vector<std::string> tNames =
                {"pressure adjoint", "temperature adjoint", "velocity adjoint", "predictor adjoint" };
            for(auto& tName : tNames)
            {
                auto tVector = aStates.vector(std::string("current ") + tName);
                aStates.vector(std::string("previous ") + tName, tVector);
            }
        }
    }

    void setPrimalStates(PrimalStates & aStates)
    {
        Plato::OrdinalType tStep = aStates.scalar("step");
        auto tCurrentVel   = Kokkos::subview(mVelocity, tStep, Kokkos::ALL());
        auto tCurrentPred  = Kokkos::subview(mPredictor, tStep, Kokkos::ALL());
        auto tCurrentTemp  = Kokkos::subview(mTemperature, tStep, Kokkos::ALL());
        auto tCurrentPress = Kokkos::subview(mPressure, tStep, Kokkos::ALL());
        aStates.vector("current velocity", tCurrentVel);
        aStates.vector("current pressure", tCurrentPress);
        aStates.vector("current temperature", tCurrentTemp);
        aStates.vector("current predictor", tCurrentPred);

        auto tPrevStep = tStep - 1;
        if (tPrevStep >= static_cast<Plato::OrdinalType>(0))
        {
            auto tPreviouVel    = Kokkos::subview(mVelocity, tPrevStep, Kokkos::ALL());
            auto tPreviousPred  = Kokkos::subview(mPredictor, tStep, Kokkos::ALL());
            auto tPreviousTemp  = Kokkos::subview(mTemperature, tPrevStep, Kokkos::ALL());
            auto tPreviousPress = Kokkos::subview(mPressure, tPrevStep, Kokkos::ALL());
            aStates.vector("previous velocity", tPreviouVel);
            aStates.vector("previous predictor", tPreviousPred);
            aStates.vector("previous pressure", tPreviousPress);
            aStates.vector("previous temperature", tPreviousTemp);
        }
        else
        {
            auto tLength = mPressure.extent(1);
            Plato::ScalarVector tPreviousPress("previous pressure", tLength);
            aStates.vector("previous pressure", tPreviousPress);
            tLength = mTemperature.extent(1);
            Plato::ScalarVector tPreviousTemp("previous temperature", tLength);
            aStates.vector("previous temperature", tPreviousTemp);
            tLength = mVelocity.extent(1);
            Plato::ScalarVector tPreviousVel("previous velocity", tLength);
            aStates.vector("previous velocity", tPreviousVel);
            tLength = mPredictor.extent(1);
            Plato::ScalarVector tPreviousPred("previous predictor", tLength);
            aStates.vector("previous previous predictor", tPreviousPred);
        }
    }

    void updateVelocity
    (const Plato::ScalarVector & aControl,
           PrimalStates        & aStates)
    {
        aStates.function("vector function");

        // calculate current residual and jacobian matrix
        auto tResidualVelocity = mVelocityResidual.value(aStates);
        auto tJacobianVelocity = mVelocityResidual.gradientCurrentVel(aStates);

        // solve velocity equation (consistent or mass lumped)
        Plato::ScalarVector tDeltaVelocity("increment", tResidualVelocity.size());
        Plato::blas1::fill(static_cast<Plato::Scalar>(0.0), tDeltaVelocity);
        mVectorFieldSolver->solve(*tJacobianVelocity, tDeltaVelocity, tResidualVelocity);

        // update velocity
        auto tCurrentVelocity  = aStates.vector("current velocity");
        auto tPreviousVelocity = aStates.vector("previous velocity");
        Plato::blas1::copy(tPreviousVelocity, tCurrentVelocity);
        Plato::blas1::axpy(1.0, tDeltaVelocity, tCurrentVelocity);
    }

    void updatePredictor
    (const Plato::ScalarVector & aControl,
           PrimalStates        & aStates)
    {
        aStates.function("vector function");

        // calculate current residual and jacobian matrix
        auto tResidualPredictor = mPredictorResidual.value(aStates);
        auto tJacobianPredictor = mPredictorResidual.gradientPredictor(aStates);

        // solve predictor equation (consistent or mass lumped)
        Plato::ScalarVector tDeltaPredictor("increment", tResidualPredictor.size());
        Plato::blas1::fill(static_cast<Plato::Scalar>(0.0), tDeltaPredictor);
        mVectorFieldSolver->solve(*tJacobianPredictor, tDeltaPredictor, tResidualPredictor);

        // update current predictor
        auto tCurrentPredictor  = aStates.vector("current predictor");
        auto tPreviousPredictor = aStates.vector("previous predictor");
        Plato::blas1::copy(tPreviousPredictor, tCurrentPredictor);
        Plato::blas1::axpy(1.0, tDeltaPredictor, tCurrentPredictor);
    }

    void updatePressure
    (const Plato::ScalarVector & aControl,
           PrimalStates        & aStates)
    {
        aStates.function("scalar function");

        // calculate current residual and jacobian matrix
        auto tResidualPressure = mPressureResidual.value(aStates);
        auto tJacobianPressure = mPressureResidual.gradientCurrentPress(aStates);

        // solve mass equation (consistent or mass lumped)
        Plato::ScalarVector tDeltaPressure("increment", tResidualPressure.size());
        Plato::blas1::fill(static_cast<Plato::Scalar>(0.0), tDeltaPressure);
        mScalarFieldSolver->solve(*tJacobianPressure, tDeltaPressure, tResidualPressure);

        // update pressure
        auto tCurrentPressure = aStates.vector("current pressure");
        auto tPreviousPressure = aStates.vector("previous pressure");
        Plato::blas1::copy(tPreviousPressure, tCurrentPressure);
        Plato::blas1::axpy(1.0, tDeltaPressure, tCurrentPressure);
    }

    void updateTemperature
    (const Plato::ScalarVector & aControl,
           PrimalStates        & aStates)
    {
        aStates.function("scalar function");

        // calculate current residual and jacobian matrix
        auto tResidualTemperature = mTemperatureResidual.value(aStates);
        auto tJacobianTemperature = mTemperatureResidual.gradientCurrentTemp(aStates);

        // solve energy equation (consistent or mass lumped)
        Plato::ScalarVector tDeltaTemperature("increment", tResidualTemperature.size());
        Plato::blas1::fill(static_cast<Plato::Scalar>(0.0), tDeltaTemperature);
        mScalarFieldSolver->solve(*tJacobianTemperature, tDeltaTemperature, tResidualTemperature);

        // update temperature
        auto tCurrentTemperature  = aStates.vector("current temperature");
        auto tPreviousTemperature = aStates.vector("previous temperature");
        Plato::blas1::copy(tPreviousTemperature, tCurrentTemperature);
        Plato::blas1::axpy(1.0, tDeltaTemperature, tCurrentTemperature);
    }

    void calculatePredictorAdjoint
    (const Plato::ScalarVector & aControl,
     const PrimalStates        & aCurrentStates,
           DualStates          & aDualStates)
    {

        auto tCurrentVelocityAdjoint = aDualStates.vector("current velocity adjoint");
        auto tGradResVelWrtPredictor = mVelocityResidual.gradientPredictor(aControl, aCurrentStates);
        Plato::ScalarVector tRHS("right hand side", tCurrentVelocityAdjoint.size());
        Plato::MatrixTimesVectorPlusVector(tGradResVelWrtPredictor, tCurrentVelocityAdjoint, tRHS);

        auto tCurrentPressureAdjoint = aDualStates.vector("current pressure adjoint");
        auto tGradResPressWrtPredictor = mPressureResidual.gradientPredictor(aControl, aCurrentStates);
        Plato::MatrixTimesVectorPlusVector(tGradResPressWrtPredictor, tCurrentPressureAdjoint, tRHS);
        Plato::blas1::scale(-1.0, tRHS);

        auto tCurrentPredictorAdjoint = aDualStates.vector("current predictor adjoint");
        Plato::blas1::fill(0.0, tCurrentPredictorAdjoint);
        auto tJacobianPredictor = mPredictorResidual.gradientPredictor(aControl, aCurrentStates);
        mVectorFieldSolver->solve(*tJacobianPredictor, tCurrentPredictorAdjoint, tRHS);
    }

    void calculatePressureAdjoint
    (const std::string         & aName,
     const Plato::ScalarVector & aControl,
     const PrimalStates        & aCurrentStates,
     const PrimalStates        & aPreviousStates,
           DualStates          & aDualStates)
    {
        auto tRHS = mCriteria[aName]->gradientCurrentPress(aControl, aCurrentStates);

        auto tGradResVelWrtCurPress = mVelocityResidual.gradientCurrentPress(aControl, aCurrentStates);
        auto tCurrentVelocityAdjoint = aDualStates.vector("current velocity adjoint");
        Plato::MatrixTimesVectorPlusVector(tGradResVelWrtCurPress, tCurrentVelocityAdjoint, tRHS);

        auto tGradResPressWrtPrevPress = mPressureResidual.gradientPreviousPress(aControl, aPreviousStates);
        auto tPrevPressureAdjoint = aDualStates.vector("previous pressure adjoint");
        Plato::MatrixTimesVectorPlusVector(tGradResPressWrtPrevPress, tPrevPressureAdjoint, tRHS);

        auto tGradResVelWrtPrevPress = mVelocityResidual.gradientPreviousPress(aControl, aPreviousStates);
        auto tPrevVelocityAdjoint = aDualStates.vector("previous velocity adjoint");
        Plato::MatrixTimesVectorPlusVector(tGradResVelWrtPrevPress, tPrevVelocityAdjoint, tRHS);

        auto tGradResPredWrtPrevPress = mPredictorResidual.gradientPreviousPress(aControl, aPreviousStates);
        auto tPrevPredictorAdjoint = aDualStates.vector("previous predictor adjoint");
        Plato::MatrixTimesVectorPlusVector(tGradResPredWrtPrevPress, tPrevPredictorAdjoint, tRHS);
        Plato::blas1::scale(-1.0, tRHS);

        auto tCurrentPressAdjoint = aDualStates.vector("current pressure adjoint");
        Plato::blas1::fill(0.0, tCurrentPressAdjoint);
        auto tJacobianPressure = mPressureResidual.gradientCurrentPress(aControl, aCurrentStates);
        mScalarFieldSolver->solve(*tJacobianPressure, tCurrentPressAdjoint, tRHS);
    }

    void calculateTemperatureAdjoint
    (const std::string         & aName,
     const Plato::ScalarVector & aControl,
     const PrimalStates        & aCurrentStates,
     const PrimalStates        & aPreviousStates,
           DualStates          & aDualStates)
    {
        auto tRHS = mCriteria[aName]->gradientCurrentTemp(aControl, aCurrentStates);

        auto tGradResPredWrtPrevTemp = mPredictorResidual.gradientPreviousTemp(aControl, aPreviousStates);
        auto tPrevPredictorAdjoint = aDualStates.vector("previous predictor adjoint");
        Plato::MatrixTimesVectorPlusVector(tGradResPredWrtPrevTemp, tPrevPredictorAdjoint, tRHS);

        auto tGradResTempWrtPrevTemp = mTemperatureResidual.gradientPreviousTemp(aControl, aPreviousStates);
        auto tPrevTempAdjoint = aDualStates.vector("previous temperature adjoint");
        Plato::MatrixTimesVectorPlusVector(tGradResTempWrtPrevTemp, tPrevTempAdjoint, tRHS);
        Plato::blas1::scale(-1.0, tRHS);

        auto tCurrentTempAdjoint = aDualStates.vector("current temperature adjoint");
        Plato::blas1::fill(0.0, tCurrentTempAdjoint);
        auto tJacobianTemperature = mTemperatureResidual.gradientCurrentTemp(aControl, aCurrentStates);
        mScalarFieldSolver->solve(*tJacobianTemperature, tCurrentTempAdjoint, tRHS);
    }

    void calculateVelocityAdjoint
    (const std::string         & aName,
     const Plato::ScalarVector & aControl,
     const PrimalStates        & aCurrentStates,
     const PrimalStates        & aPreviousStates,
           DualStates          & aDualStates)
    {
        auto tRHS = mCriteria[aName]->gradientCurrentVel(aControl, aCurrentStates);

        auto tGradResPredWrtPrevVel = mPredictorResidual.gradientPreviousVel(aControl, aPreviousStates);
        auto tPrevPredictorAdjoint = aDualStates.vector("previous predictor adjoint");
        Plato::MatrixTimesVectorPlusVector(tGradResPredWrtPrevVel, tPrevPredictorAdjoint, tRHS);

        auto tGradResVelWrtPrevVel = mVelocityResidual.gradientPreviousVel(aControl, aPreviousStates);
        auto tPrevVelocityAdjoint = aDualStates.vector("previous velocity adjoint");
        Plato::MatrixTimesVectorPlusVector(tGradResVelWrtPrevVel, tPrevVelocityAdjoint, tRHS);

        auto tGradResPressWrtPrevVel = mPressureResidual.gradientPreviousVel(aControl, aPreviousStates);
        auto tPrevPressureAdjoint = aDualStates.vector("previous pressure adjoint");
        Plato::MatrixTimesVectorPlusVector(tGradResPressWrtPrevVel, tPrevPressureAdjoint, tRHS);

        auto tGradResTempWrtPrevVel = mTemperatureResidual.gradientPreviousVel(aControl, aPreviousStates);
        auto tPrevTemperatureAdjoint = aDualStates.vector("previous temperature adjoint");
        Plato::MatrixTimesVectorPlusVector(tGradResTempWrtPrevVel, tPrevTemperatureAdjoint, tRHS);
        Plato::blas1::scale(-1.0, tRHS);

        auto tCurrentVelocityAdjoint = aDualStates.vector("current velocity adjoint");
        Plato::blas1::fill(0.0, tCurrentVelocityAdjoint);
        auto tJacobianVelocity = mVelocityResidual.gradientCurrentVel(aControl, aCurrentStates);
        mVectorFieldSolver->solve(*tJacobianVelocity, tCurrentVelocityAdjoint, tRHS);
    }

    void calculateGradientControl
    (const std::string         & aName,
     const Plato::ScalarVector & aControl,
     const PrimalStates        & aCurrentStates,
     const DualStates          & aDualStates,
           Plato::ScalarVector & aTotalDerivative)
    {
        auto tGradCriterionWrtControl = mCriteria[aName]->gradientControl(aControl, aCurrentStates);

        auto tCurrentPredictorAdjoint = aDualStates.vector("current predictor adjoint");
        auto tGradResPredWrtControl = mPredictorResidual.gradientControl(aControl, aCurrentStates);
        Plato::MatrixTimesVectorPlusVector(tGradResPredWrtControl, tCurrentPredictorAdjoint, tGradCriterionWrtControl);

        auto tCurrentPressureAdjoint = aDualStates.vector("current pressure adjoint");
        auto tGradResPressWrtControl = mPressureResidual.gradientControl(aControl, aCurrentStates);
        Plato::MatrixTimesVectorPlusVector(tGradResPressWrtControl, tCurrentPressureAdjoint, tGradCriterionWrtControl);

        auto tCurrentTemperatureAdjoint = aDualStates.vector("current temperature adjoint");
        auto tGradResTempWrtControl = mTemperatureResidual.gradientControl(aControl, aCurrentStates);
        Plato::MatrixTimesVectorPlusVector(tGradResTempWrtControl, tCurrentTemperatureAdjoint, tGradCriterionWrtControl);

        auto tCurrentVelocityAdjoint = aDualStates.vector("current velocity adjoint");
        auto tGradResVelWrtControl = mVelocityResidual.gradientControl(aControl, aCurrentStates);
        Plato::MatrixTimesVectorPlusVector(tGradResVelWrtControl, tCurrentVelocityAdjoint, tGradCriterionWrtControl);

        Plato::blas1::axpy(1.0, tGradCriterionWrtControl, aTotalDerivative);
    }

    void calculateGradientConfig
    (const std::string         & aName,
     const Plato::ScalarVector & aControl,
     const PrimalStates        & aCurrentStates,
     const DualStates          & aDualStates,
           Plato::ScalarVector & aTotalDerivative)
    {
        auto tGradCriterionWrtConfig = mCriteria[aName]->gradientConfig(aControl, aCurrentStates);

        auto tCurrentPredictorAdjoint = aDualStates.vector("current predictor adjoint");
        auto tGradResPredWrtConfig = mPredictorResidual.gradientConfig(aControl, aCurrentStates);
        Plato::MatrixTimesVectorPlusVector(tGradResPredWrtConfig, tCurrentPredictorAdjoint, tGradCriterionWrtConfig);

        auto tCurrentPressureAdjoint = aDualStates.vector("current pressure adjoint");
        auto tGradResPressWrtConfig = mPressureResidual.gradientConfig(aControl, aCurrentStates);
        Plato::MatrixTimesVectorPlusVector(tGradResPressWrtConfig, tCurrentPressureAdjoint, tGradCriterionWrtConfig);

        auto tCurrentTemperatureAdjoint = aDualStates.vector("current temperature adjoint");
        auto tGradResTempWrtConfig = mTemperatureResidual.gradientConfig(aControl, aCurrentStates);
        Plato::MatrixTimesVectorPlusVector(tGradResTempWrtConfig, tCurrentTemperatureAdjoint, tGradCriterionWrtConfig);

        auto tCurrentVelocityAdjoint = aDualStates.vector("current velocity adjoint");
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


class WorkSetBase
{
public:
    virtual ~WorkSetBase() = 0;
};
inline WorkSetBase::~WorkSetBase(){}

template<class Type>
class WorkSet : public WorkSetBase
{
public:
    explicit WorkSet(const Type &aData) : mData(aData) {}
    WorkSet() {}
    Type mData;
};

template<class Type>
inline Type workset(std::shared_ptr<Plato::WorkSetBase> & aInput)
{
    return (dynamic_cast<Plato::WorkSet<Type>&>(aInput.operator*()).mData);
}

//todo: add free function to set use case specific worksets based on the scalar evaluation types

class WorkSets
{
private:
    std::unordered_map<std::string, std::shared_ptr<Plato::WorkSetBase>> mData;

public:
    WorkSets() {}
    void set(const std::string & aName, const std::shared_ptr<Plato::WorkSetBase> & aData)
    {
        mData[aName] = aData;
    }

    std::shared_ptr<Plato::WorkSetBase> get(const std::string & aName) const
    {
        auto tItr = mData.find(aName);
        if(tItr != mData.end())
        {
            return tItr->second;
        }
        else
        {
            THROWERR(std::string("Did not find 'WorkSetBase' with tag '") + aName + "'.")
        }
    }
};

}
//namespace Plato


namespace ComputationalFluidDynamicsTests
{

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, IsValidFunction)
{
    // 1. test throw
    TEST_THROW(Plato::is_valid_function("some function"), std::runtime_error);

    // 2. test scalar function
    auto tOutput = Plato::is_valid_function("scalar function");
    TEST_COMPARE(tOutput, ==, "scalar function");

    // 2. test vector function
    tOutput = Plato::is_valid_function("vector function");
    TEST_COMPARE(tOutput, ==, "vector function");

}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, SidesetNames)
{
    Teuchos::RCP<Teuchos::ParameterList> tParams =
    Teuchos::getParametersFromXmlString(
        "<ParameterList  name='Natural Boundary Conditions'>"
        "  <ParameterList  name='Traction Vector Boundary Condition 1'>"
        "    <Parameter  name='Type'     type='string'        value='Uniform'/>"
        "    <Parameter  name='Values'   type='Array(double)' value='{0.0, -3.0e3, 0.0}'/>"
        "    <Parameter  name='Sides'    type='string'        value='ss_1'/>"
        "  </ParameterList>"
        "  <ParameterList  name='Traction Vector Boundary Condition 2'>"
        "    <Parameter  name='Type'     type='string'        value='Uniform'/>"
        "    <Parameter  name='Values'   type='Array(double)' value='{0.0, -3.0e3, 0.0}'/>"
        "    <Parameter  name='Sides'    type='string'        value='ss_2'/>"
        "  </ParameterList>"
        "</ParameterList>"
    );

    auto tBCs = tParams->sublist("Natural Boundary Conditions");
    auto tOutput = Plato::sideset_names(tBCs);

    std::vector<std::string> tGold = {"ss_1", "ss_2"};
    for(auto& tName : tOutput)
    {
        auto tIndex = &tName - &tOutput[0];
        TEST_COMPARE(tName, ==, tGold[tIndex]);
    }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ParseDimensionlessProperty)
{
    Teuchos::RCP<Teuchos::ParameterList> tParams =
    Teuchos::getParametersFromXmlString(
        "<ParameterList  name='Plato Problem'>"
        "  <ParameterList  name='Dimensionless Properties'>"
        "    <Parameter  name='Prandtl'   type='double'        value='2.1'/>"
        "    <Parameter  name='Grashof'   type='Array(double)' value='{0.0, 1.5, 0.0}'/>"
        "    <Parameter  name='Darcy'     type='double'        value='2.2'/>"
        "  </ParameterList>"
        "</ParameterList>"
    );

    // Prandtl #
    auto tScalarOutput = Plato::parse_dimensionless_property<Plato::Scalar>(tParams.operator*(), "Prandtl");
    auto tTolerance = 1e-6;
    TEST_FLOATING_EQUALITY(tScalarOutput, 2.1, tTolerance);

    // Darcy #
    tScalarOutput = Plato::parse_dimensionless_property<Plato::Scalar>(tParams.operator*(), "Darcy");
    TEST_FLOATING_EQUALITY(tScalarOutput, 2.2, tTolerance);

    // Grashof #
    auto tArrayOutput = Plato::parse_dimensionless_property<Teuchos::Array<Plato::Scalar>>(tParams.operator*(), "Grashof");
    TEST_EQUALITY(3, tArrayOutput.size());
    TEST_FLOATING_EQUALITY(tArrayOutput[0], 0.0, tTolerance);
    TEST_FLOATING_EQUALITY(tArrayOutput[1], 1.5, tTolerance);
    TEST_FLOATING_EQUALITY(tArrayOutput[2], 0.0, tTolerance);
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, SolutionsStruct)
{
    Plato::Solutions tSolution;
    constexpr Plato::OrdinalType tNumTimeSteps = 2;

    // set velocity
    constexpr Plato::OrdinalType tNumVelDofs = 12;
    Plato::ScalarMultiVector tGoldVel("velocity", tNumTimeSteps, tNumVelDofs);
    auto tHostGoldVel = Kokkos::create_mirror(tGoldVel);
    for(auto tStep = 0; tStep < tNumTimeSteps; tStep++)
    {
        for(auto tDof = 0; tDof < tNumVelDofs; tDof++)
        {
            tHostGoldVel(tStep, tDof) = (tStep * tNumTimeSteps) + tDof;
        }
    }
    Kokkos::deep_copy(tGoldVel, tHostGoldVel);
    tSolution.set("velocity", tGoldVel);

    // set pressure
    constexpr Plato::OrdinalType tNumPressDofs = 6;
    Plato::ScalarMultiVector tGoldPress("pressure", tNumTimeSteps, tNumPressDofs);
    auto tHostGoldPress = Kokkos::create_mirror(tGoldPress);
    for(auto tStep = 0; tStep < tNumTimeSteps; tStep++)
    {
        for(auto tDof = 0; tDof < tNumPressDofs; tDof++)
        {
            tHostGoldPress(tStep, tDof) = (tStep * tNumTimeSteps) + tDof;
        }
    }
    Kokkos::deep_copy(tGoldPress, tHostGoldPress);
    tSolution.set("pressure", tGoldPress);

    // set temperature
    constexpr Plato::OrdinalType tNumTempDofs = 6;
    Plato::ScalarMultiVector tGoldTemp("temperature", tNumTimeSteps, tNumTempDofs);
    auto tHostGoldTemp = Kokkos::create_mirror(tGoldTemp);
    for(auto tStep = 0; tStep < tNumTimeSteps; tStep++)
    {
        for(auto tDof = 0; tDof < tNumTempDofs; tDof++)
        {
            tHostGoldTemp(tStep, tDof) = (tStep * tNumTimeSteps) + tDof;
        }
    }
    Kokkos::deep_copy(tGoldTemp, tHostGoldTemp);
    tSolution.set("temperature", tGoldTemp);

    // ********** test velocity **********
    auto tTolerance = 1e-6;
    auto tVel   = tSolution.get("velocity");
    auto tHostVel = Kokkos::create_mirror(tVel);
    tHostGoldVel  = Kokkos::create_mirror(tGoldVel);
    for(auto tStep = 0; tStep < tNumTimeSteps; tStep++)
    {
        for(auto tDof = 0; tDof < tNumVelDofs; tDof++)
        {
            TEST_FLOATING_EQUALITY(tHostGoldVel(tStep, tDof), tHostVel(tStep, tDof), tTolerance);
        }
    }

    // ********** test pressure **********
    auto tPress = tSolution.get("pressure");
    auto tHostPress = Kokkos::create_mirror(tPress);
    tHostGoldPress  = Kokkos::create_mirror(tGoldPress);
    for(auto tStep = 0; tStep < tNumTimeSteps; tStep++)
    {
        for(auto tDof = 0; tDof < tNumPressDofs; tDof++)
        {
            TEST_FLOATING_EQUALITY(tHostGoldPress(tStep, tDof), tHostPress(tStep, tDof), tTolerance);
        }
    }

    // ********** test temperature **********
    auto tTemp  = tSolution.get("temperature");
    auto tHostTemp = Kokkos::create_mirror(tTemp);
    tHostGoldTemp  = Kokkos::create_mirror(tGoldTemp);
    for(auto tStep = 0; tStep < tNumTimeSteps; tStep++)
    {
        for(auto tDof = 0; tDof < tNumTempDofs; tDof++)
        {
            TEST_FLOATING_EQUALITY(tHostGoldTemp(tStep, tDof), tHostTemp(tStep, tDof), tTolerance);
        }
    }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, StatesStruct)
{
    Plato::FluidMechanics::States tStates;
    TEST_COMPARE(tStates.isScalarMapEmpty(), ==, true);
    TEST_COMPARE(tStates.isVectorMapEmpty(), ==, true);

    // set function type
    tStates.function("vector function");
    TEST_COMPARE(tStates.function(), ==, "vector function");
    TEST_COMPARE(tStates.function(), !=, "scalar function");

    // set time step index
    tStates.scalar("step", 1);
    TEST_COMPARE(tStates.isScalarMapEmpty(), ==, false);

    // set velocity
    constexpr Plato::OrdinalType tNumVelDofs = 12;
    Plato::ScalarVector tGoldVel("velocity", tNumVelDofs);
    auto tHostGoldVel = Kokkos::create_mirror(tGoldVel);
    for(auto tDof = 0; tDof < tNumVelDofs; tDof++)
    {
        tHostGoldVel(tDof) = tDof;
    }
    Kokkos::deep_copy(tGoldVel, tHostGoldVel);
    tStates.vector("velocity", tGoldVel);
    TEST_COMPARE(tStates.isVectorMapEmpty(), ==, false);

    // set pressure
    constexpr Plato::OrdinalType tNumPressDofs = 6;
    Plato::ScalarVector tGoldPress("pressure", tNumPressDofs);
    auto tHostGoldPress = Kokkos::create_mirror(tGoldPress);
    for(auto tDof = 0; tDof < tNumPressDofs; tDof++)
    {
        tHostGoldPress(tDof) = tDof;
    }
    Kokkos::deep_copy(tGoldPress, tHostGoldPress);
    tStates.vector("pressure", tGoldPress);

    // test empty funciton
    TEST_COMPARE(tStates.empty("velocity"), ==, false);
    TEST_COMPARE(tStates.empty("temperature"), ==, true);
    TEST_COMPARE(tStates.empty("pressure", "vector"), ==, false);
    TEST_COMPARE(tStates.empty("step", "scalar"), ==, false);
    TEST_COMPARE(tStates.empty("time step", "scalar"), ==, true);

    // test metadata
    auto tTolerance = 1e-6;
    TEST_FLOATING_EQUALITY(1.0, tStates.scalar("step"), tTolerance);

    auto tVel  = tStates.vector("velocity");
    auto tHostVel = Kokkos::create_mirror(tVel);
    tHostGoldVel  = Kokkos::create_mirror(tGoldVel);
    for(auto tDof = 0; tDof < tNumVelDofs; tDof++)
    {
        TEST_FLOATING_EQUALITY(tHostGoldVel(tDof), tHostVel(tDof), tTolerance);
    }

    auto tPress  = tStates.vector("pressure");
    auto tHostPress = Kokkos::create_mirror(tPress);
    tHostGoldPress  = Kokkos::create_mirror(tGoldPress);
    for(auto tDof = 0; tDof < tNumPressDofs; tDof++)
    {
        TEST_FLOATING_EQUALITY(tHostGoldPress(tDof), tHostPress(tDof), tTolerance);
    }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, Plato_FluidMechanics_WorkSets)
{
    constexpr Plato::OrdinalType tNumCells = 2;
    constexpr Plato::OrdinalType tSpaceDim = 3;
    using PhysicsT = Plato::IncompressibleFluids<tSpaceDim>;
    using ResidualEvalT = typename Plato::FluidMechanics::Evaluation<typename PhysicsT::SimplexT>::Residual;
    Plato::FluidMechanics::WorkSets<PhysicsT, ResidualEvalT> tWorksets(tNumCells);
    TEST_EQUALITY(tNumCells, tWorksets.numCells());

    // test scalar fields
    constexpr Plato::OrdinalType tNumNodesPerCell = 4;
    TEST_EQUALITY(tNumCells, tWorksets.artificialCompress().extent(0));
    TEST_EQUALITY(tNumNodesPerCell, tWorksets.artificialCompress().extent(1));
    TEST_EQUALITY(tNumCells, tWorksets.timeStep().extent(0));
    TEST_EQUALITY(tNumNodesPerCell, tWorksets.timeStep().extent(1));
    TEST_EQUALITY(tNumCells, tWorksets.control().extent(0));
    TEST_EQUALITY(tNumNodesPerCell, tWorksets.control().extent(1));
    TEST_EQUALITY(tNumCells, tWorksets.previousPressure().extent(0));
    TEST_EQUALITY(tNumNodesPerCell, tWorksets.previousPressure().extent(1));
    TEST_EQUALITY(tNumCells, tWorksets.currentPressure().extent(0));
    TEST_EQUALITY(tNumNodesPerCell, tWorksets.currentPressure().extent(1));
    TEST_EQUALITY(tNumCells, tWorksets.previousTemperature().extent(0));
    TEST_EQUALITY(tNumNodesPerCell, tWorksets.previousTemperature().extent(1));
    TEST_EQUALITY(tNumCells, tWorksets.currentTemperature().extent(0));
    TEST_EQUALITY(tNumNodesPerCell, tWorksets.currentTemperature().extent(1));

    // test vector fields
    constexpr Plato::OrdinalType tNumVelDofsPerCell = tSpaceDim * tNumNodesPerCell;
    TEST_EQUALITY(tNumCells, tWorksets.predictor().extent(0));
    TEST_EQUALITY(tNumVelDofsPerCell, tWorksets.predictor().extent(1));
    TEST_EQUALITY(tNumCells, tWorksets.previousVelocity().extent(0));
    TEST_EQUALITY(tNumVelDofsPerCell, tWorksets.previousVelocity().extent(1));
    TEST_EQUALITY(tNumCells, tWorksets.currentVelocity().extent(0));
    TEST_EQUALITY(tNumVelDofsPerCell, tWorksets.currentVelocity().extent(1));

    // test configuration
    TEST_EQUALITY(tNumCells, tWorksets.configuration().extent(0));
    TEST_EQUALITY(tNumNodesPerCell, tWorksets.configuration().extent(1));
    TEST_EQUALITY(tSpaceDim, tWorksets.configuration().extent(2));
}

}