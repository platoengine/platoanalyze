/*
 * ComputationalFluidDynamicsTests.cpp
 *
 *  Created on: Oct 13, 2020
 */

#include <Teuchos_UnitTestHarness.hpp>
#include <Teuchos_XMLParameterListCoreHelpers.hpp>
#include <unordered_map>

#include <Omega_h_mark.hpp>
#include <Omega_h_assoc.hpp>
#include <Omega_h_shape.hpp>
#include <Omega_h_array_ops.hpp>

#include "BLAS1.hpp"
#include "BLAS2.hpp"
#include "BLAS3.hpp"
#include "Simplex.hpp"
#include "Assembly.hpp"
#include "NaturalBCs.hpp"
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

namespace blas2
{

template<Plato::OrdinalType LengthI,
         Plato::OrdinalType LengthJ,
         typename ScalarT,
         typename AViewTypeT,
         typename BViewTypeT>
DEVICE_TYPE inline void
scale(const Plato::OrdinalType & aCellOrdinal,
      const ScalarT & aScalar,
      const Plato::ScalarArray3DT<AViewTypeT> & aInputWS,
      const Plato::ScalarArray3DT<BViewTypeT> & aOutputWS)
{
    for(Plato::OrdinalType tDimI = 0; tDimI < LengthI; tDimI++)
    {
        for(Plato::OrdinalType tDimJ = 0; tDimJ < LengthJ; tDimJ++)
        {
            aOutputWS(aCellOrdinal, tDimI, tDimJ) = aScalar * aInputWS(aCellOrdinal, tDimI, tDimJ);
        }
    }
}

template<Plato::OrdinalType LengthI,
         Plato::OrdinalType LengthJ,
         typename AViewType,
         typename BViewType,
         typename CViewType>
DEVICE_TYPE inline void
dot(const Plato::OrdinalType & aCellOrdinal,
    const Plato::ScalarArray3DT<AViewType> & aTensorA,
    const Plato::ScalarArray3DT<BViewType> & aTensorB,
    const Plato::ScalarVectorT <CViewType> & aOutput)
{
    for(Plato::OrdinalType tDimI = 0; tDimI < LengthI; tDimI++)
    {
        for(Plato::OrdinalType tDimJ = 0; tDimJ < LengthJ; tDimJ++)
        {
            aOutput(aCellOrdinal) += aTensorA(aCellOrdinal, tDimI, tDimJ) * aTensorB(aCellOrdinal, tDimI, tDimJ);
        }
    }
}

}
// namespace blas3

namespace blas1
{

template<Plato::OrdinalType Length,
         typename ScalarT,
         typename AViewTypeT,
         typename BViewTypeT>
DEVICE_TYPE inline void
scale(const Plato::OrdinalType & aCellOrdinal,
      const ScalarT & aScalar,
      const Plato::ScalarMultiVectorT<AViewTypeT> & aInputWS,
      const Plato::ScalarMultiVectorT<BViewTypeT> & aOutputWS)
{
    for(Plato::OrdinalType tDim = 0; tDim < Length; tDim++)
    {
        aOutputWS(aCellOrdinal, tDim) = aScalar * aInputWS(aCellOrdinal, tDim);
    }
}

template<Plato::OrdinalType Length,
         typename AViewType,
         typename BViewType,
         typename CViewType>
DEVICE_TYPE inline void
dot(const Plato::OrdinalType & aCellOrdinal,
    const Plato::ScalarMultiVectorT<AViewType> & aVectorA,
    const Plato::ScalarMultiVectorT<BViewType> & aVectorB,
    const Plato::ScalarVectorT<CViewType>      & aOutput)
{
    for(Plato::OrdinalType tDim = 0; tDim < Length; tDim++)
    {
        aOutput(aCellOrdinal) += aVectorA(aCellOrdinal, tDim) * aVectorB(aCellOrdinal, tDim);
    }
}

}
// namespace blas1


template <typename Type>
inline void print_fad_val_values
(const Plato::ScalarVectorT<Type> & aInput,
 const std::string & aName)
{
    std::cout << "\nStart: Print ScalarVector '" << aName << "'.\n";
    const auto tLength = aInput.extent(0);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tLength), LAMBDA_EXPRESSION(const Plato::OrdinalType & aOrdinal)
    {
        printf("Input(%d) = %f\n", aOrdinal, aInput(aOrdinal).val());
    }, "print_fad_val_values");
    std::cout << "End: Print ScalarVector '" << aName << "'.\n";
}

template <Plato::OrdinalType NumNodesPerCell,
          Plato::OrdinalType NumDofsPerNode,
          typename Type>
inline void print_fad_dx_values
(const Plato::ScalarVectorT<Type> & aInput,
 const std::string & aName)
{
    std::cout << "\nStart: Print ScalarVector '" << aName << "'.\n";
    const auto tLength = aInput.extent(0);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tLength), LAMBDA_EXPRESSION(const Plato::OrdinalType & aOrdinal)
    {
        for(Plato::OrdinalType tNode=0; tNode < NumNodesPerCell; tNode++)
        {
            for(Plato::OrdinalType tDof=0; tDof < NumDofsPerNode; tDof++)
            {
                printf("Input(Cell=%d,Node=%d,Dof=%d) = %f\n", aOrdinal, tNode, tDof, aInput(aOrdinal).dx(tNode * NumDofsPerNode + tDof));
            }
        }
    }, "print_fad_dx_values");
    std::cout << "End: Print ScalarVector '" << aName << "'.\n";
}

namespace omega_h
{
template<typename ArrayT>
void print(const ArrayT & aInput, const std::string & aName)
{
    std::cout << "Start Printing Array with Name '" << aName << "'\n";
    auto tLength = aInput.size();
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tLength), LAMBDA_EXPRESSION(const Plato::OrdinalType & aOrdinal)
    {
        printf("Array(%d)=%d\n",aOrdinal,aInput[aOrdinal]);
    }, "print");
    std::cout << "Finished Printing Array with Name '" << aName << "'\n";
}
}

template<Omega_h::SetType EntitySet>
inline Omega_h::LOs
entity_ordinals
(const Omega_h::MeshSets& aMeshSets,
 const std::string& aSetName,
 bool aThrow = true)
{
    auto& tEntitySets = aMeshSets[EntitySet];
    auto tEntitySetMapIterator = tEntitySets.find(aSetName);
    if( (tEntitySetMapIterator == tEntitySets.end()) && (aThrow) )
    {
        THROWERR(std::string("DID NOT FIND NODE SET WITH NAME '") + aSetName + "'. NODE SET '"
                 + aSetName + "' IS NOT DEFINED IN INPUT MESH FILE, I.E. INPUT EXODUS FILE");
    }
    auto tFaceLids = (tEntitySetMapIterator->second);
    return tFaceLids;
}
// function node_set_face_ordinals

template<Omega_h::SetType EntitySet>
inline bool
is_entity_set_defined
(const Omega_h::MeshSets& aMeshSets,
 const std::string& aSetName)
{
    auto& tNodeSets = aMeshSets[EntitySet];
    auto tNodeSetMapItr = tNodeSets.find(aSetName);
    auto tIsNodeSetDefined = tNodeSetMapItr != tNodeSets.end() ? true : false;
    return tIsNodeSetDefined;
}

template<Omega_h::SetType EntitySet>
inline Omega_h::LOs
get_entity_ordinals
(const Omega_h::MeshSets& aMeshSets,
 const std::string& aSetName,
 bool aThrow = true)
{

    if( Plato::is_entity_set_defined<EntitySet>(aMeshSets, aSetName) )
    {
        auto tEntityFaceOrdinals = Plato::entity_ordinals<EntitySet>(aMeshSets, aSetName);
        return tEntityFaceOrdinals;
    }
    else
    {
        THROWERR(std::string("Entity set, i.e. side or node set, with name '") + aSetName + "' is not defined.")
    }
}

inline Plato::OrdinalType
get_num_entities
(const Omega_h::Int aEntityDim,
 const Omega_h::Mesh & aMesh)
{
    if(aEntityDim == Omega_h::VERT)
    {
        return aMesh.nverts();
    }
    else if(aEntityDim == Omega_h::EDGE)
    {
        return aMesh.nedges();
    }
    else if(aEntityDim == Omega_h::FACE)
    {
        return aMesh.nfaces();
    }
    else if(aEntityDim == Omega_h::REGION)
    {
        return aMesh.nelems();
    }
    else
    {
        THROWERR("Entity is not supported. Supported entities: Omega_h::VERT, Omega_h::EDGE, Omega_h::FACE, and Omega_h::REGION")
    }
}

template
<Omega_h::Int EntityDim,
 Omega_h::SetType EntitySet>
inline Omega_h::LOs
entities_on_non_prescribed_boundary
(const std::vector<std::string> & aEntitySetNames,
       Omega_h::Mesh            & aMesh,
       Omega_h::MeshSets        & aMeshSets)
{
    // returns all the boundary faces, excluding faces within the domain
    auto tEntitiesAreOnNonPrescribedBoundary = Omega_h::mark_by_class_dim(&aMesh, EntityDim, EntityDim);
    // loop over all the side sets to get non-prescribed boundary faces
    auto tNumEntities = Plato::get_num_entities(EntityDim, aMesh);
    for(auto& tEntitySetName : aEntitySetNames)
    {
        // return entity ids on prescribed side set
        auto tEntitiesOnPrescribedBoundary = Plato::get_entity_ordinals<EntitySet>(aMeshSets, tEntitySetName);
        // return boolean array (entity on prescribed side set=1, entity not on prescribed side set=0)
        auto tEntitiesAreOnPrescribedBoundary = Omega_h::mark_image(tEntitiesOnPrescribedBoundary, tNumEntities);
        // return boolean array with 1's for all entities not on prescribed side set and 0's otherwise
        auto tEntitiesAreNotOnPrescribedBoundary = Omega_h::invert_marks(tEntitiesAreOnPrescribedBoundary);
        // return boolean array (entity on the non-prescribed boundary=1, entity not on the non-prescribed boundary=0)
        tEntitiesAreOnNonPrescribedBoundary = Omega_h::land_each(tEntitiesAreOnNonPrescribedBoundary, tEntitiesAreNotOnPrescribedBoundary);
    }
    // return identification numbers of all the entities on the non-prescribed boundary
    auto tIDsOfEntitiesOnNonPrescribedBoundary = Omega_h::collect_marked(tEntitiesAreOnNonPrescribedBoundary);
    return tIDsOfEntitiesOnNonPrescribedBoundary;
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
            THROWERR("sideset_names: Parameter list block is not valid.  Expect lists only.")
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

template <typename T>
inline std::vector<T>
parse_array
(const std::string & aTag,
 const Teuchos::ParameterList & aInputs)
{
    if(!aInputs.isParameter(aTag))
    {
        std::vector<T> tOutput;
        return tOutput;
    }
    auto tSideSets = aInputs.get< Teuchos::Array<T> >(aTag);

    auto tLength = tSideSets.size();
    std::vector<T> tOutput(tLength);
    for(auto & tName : tOutput)
    {
        auto tIndex = &tName - &tOutput[0];
        tOutput[tIndex] = tSideSets[tIndex];
    }
    return tOutput;
}

template <typename T>
inline T parse_parameter
(const std::string            & aTag,
 const std::string            & aBlock,
 const Teuchos::ParameterList & aInputs)
{
    if( !aInputs.isSublist(aBlock) )
    {
        THROWERR(std::string("Parameter Sublist '") + aBlock + "' within Paramater List '" 
            + aInputs.name() + "' is not defined.")
    }
    auto tSublist = aInputs.sublist(aBlock);

    if( !tSublist.isParameter(aTag) )
    {
        THROWERR(std::string("Parameter with '") + aTag + "' is not defined in Parameter Sublist with name '" + aBlock + "'.")
    }
    auto tOutput = tSublist.get<T>(aTag);
    return tOutput;
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

    // optimizable quantities of interest
    static constexpr Plato::OrdinalType mNumConfigDofsPerNode  = mNumSpatialDims; /*!< number of configuration degrees of freedom per node */
    static constexpr Plato::OrdinalType mNumControlDofsPerNode = NumControls;     /*!< number of controls per node */
    static constexpr Plato::OrdinalType mNumConfigDofsPerCell  = mNumConfigDofsPerNode * mNumNodesPerCell;  /*!< number of configuration degrees of freedom per cell */
    static constexpr Plato::OrdinalType mNumControlDofsPerCell = mNumControlDofsPerNode * mNumNodesPerCell; /*!< number of controls per cell */

    // physical quantities of interest
    static constexpr Plato::OrdinalType mNumMassDofsPerNode     = 1; /*!< number of continuity degrees of freedom per node */
    static constexpr Plato::OrdinalType mNumMassDofsPerCell     = mNumMassDofsPerNode * mNumNodesPerCell; /*!< number of continuity degrees of freedom per cell */
    static constexpr Plato::OrdinalType mNumEnergyDofsPerNode   = 1; /*!< number energy degrees of freedom per node */
    static constexpr Plato::OrdinalType mNumEnergyDofsPerCell   = mNumEnergyDofsPerNode * mNumNodesPerCell; /*!< number of energy degrees of freedom per cell */
    static constexpr Plato::OrdinalType mNumMomentumDofsPerNode = mNumSpatialDims; /*!< number of momentum degrees of freedom per node */
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
            THROWERR(std::string("Solution with tag '") + aTag + "' is not defined in the solution map.")
        }
        return tItr->second;
    }
};
// struct Solutions


class MetaDataBase
{
public:
    virtual ~MetaDataBase() = 0;
};
inline MetaDataBase::~MetaDataBase(){}

template<class Type>
class MetaData : public MetaDataBase
{
public:
    explicit MetaData(const Type &aData) : mData(aData) {}
    MetaData() {}
    Type mData;
};

template<class Type>
inline Type metadata(const std::shared_ptr<Plato::MetaDataBase> & aInput)
{
    return (dynamic_cast<Plato::MetaData<Type>&>(aInput.operator*()).mData);
}

struct WorkSets
{
private:
    std::unordered_map<std::string, std::shared_ptr<Plato::MetaDataBase>> mData;

public:
    WorkSets(){}

    void set(const std::string & aName, const std::shared_ptr<Plato::MetaDataBase> & aData)
    {
        auto tLowerKey = Plato::tolower(aName);
        mData[tLowerKey] = aData;
    }

    const std::shared_ptr<Plato::MetaDataBase> & get(const std::string & aName) const
    {
        auto tLowerKey = Plato::tolower(aName);
        auto tItr = mData.find(tLowerKey);
        if(tItr != mData.end())
        {
            return tItr->second;
        }
        else
        {
            THROWERR(std::string("Did not find 'MetaData' with tag '") + aName + "'.")
        }
    }

    std::vector<std::string> tags() const
    {
        std::vector<std::string> tOutput;
        for(auto& tPair : mData)
        {
            tOutput.push_back(tPair.first);
        }
        return tOutput;
    }

    bool defined(const std::string & aTag) const
    {
        auto tLowerKey = Plato::tolower(aTag);
        auto tItr = mData.find(tLowerKey);
        auto tFound = tItr != mData.end();
        if(tFound)
        { return true; }
        else
        { return false; }
    }
};

template <typename PhysicsT>
struct LocalOrdinalMaps
{
    Plato::NodeCoordinate<PhysicsT::SimplexT::mNumSpatialDims> mNodeCoordinate;
    Plato::VectorEntryOrdinal<PhysicsT::SimplexT::mNumSpatialDims, 1 /*scalar dofs per node*/>                 mScalarStateOrdinalMap;
    Plato::VectorEntryOrdinal<PhysicsT::SimplexT::mNumSpatialDims, PhysicsT::SimplexT::mNumSpatialDims>        mVectorStateOrdinalMap;
    Plato::VectorEntryOrdinal<PhysicsT::SimplexT::mNumSpatialDims, PhysicsT::SimplexT::mNumControlDofsPerNode> mControlOrdinalMap;

    LocalOrdinalMaps(Omega_h::Mesh & aMesh) :
        mNodeCoordinate(&aMesh),
        mScalarStateOrdinalMap(&aMesh),
        mVectorStateOrdinalMap(&aMesh),
        mControlOrdinalMap(&aMesh)
    { return; }
};

struct Variables
{
private:
    std::unordered_map<std::string, Plato::Scalar> mScalars;
    std::unordered_map<std::string, Plato::ScalarVector> mVectors;

public:
    Plato::Scalar scalar(const std::string& aTag) const
    {
        auto tLowerTag = Plato::tolower(aTag);
        auto tItr = mScalars.find(tLowerTag);
        if(tItr == mScalars.end())
        {
            THROWERR(std::string("Scalar with tag '") + aTag + "' is not defined in the variables map.")
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
            THROWERR(std::string("Vector with tag '") + aTag + "' is not defined in the variables map.")
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

    bool defined(const std::string & aTag) const
    {
        auto tLowerTag = Plato::tolower(aTag);
        auto tScalarMapItr = mScalars.find(tLowerTag);
        auto tFoundScalarTag = tScalarMapItr != mScalars.end();
        auto tVectorMapItr = mVectors.find(tLowerTag);
        auto tFoundVectorTag = tVectorMapItr != mVectors.end();

        if(tFoundScalarTag || tFoundVectorTag)
        { return true; }
        else
        { return false; }
    }
};
typedef Variables Dual;
typedef Variables Primal;


namespace FluidMechanics
{

template<typename SimplexPhysics>
struct SimplexFadTypes
{
    using ConfigFad   = Sacado::Fad::SFad<Plato::Scalar, SimplexPhysics::mNumConfigDofsPerCell>;
    using ControlFad  = Sacado::Fad::SFad<Plato::Scalar, SimplexPhysics::mNumNodesPerCell>;
    using MassFad     = Sacado::Fad::SFad<Plato::Scalar, SimplexPhysics::mNumMassDofsPerCell>;
    using EnergyFad   = Sacado::Fad::SFad<Plato::Scalar, SimplexPhysics::mNumEnergyDofsPerCell>;
    using MomentumFad = Sacado::Fad::SFad<Plato::Scalar, SimplexPhysics::mNumMomentumDofsPerCell>;
};

// is_fad<TypesT, T>::value is true if T is of any AD type defined TypesT.
//
template <typename SimplexFadTypesT, typename ScalarType>
struct is_fad {
  static constexpr bool value = std::is_same< ScalarType, typename SimplexFadTypesT::MassFad     >::value ||
                                std::is_same< ScalarType, typename SimplexFadTypesT::ControlFad  >::value ||
                                std::is_same< ScalarType, typename SimplexFadTypesT::ConfigFad   >::value ||
                                std::is_same< ScalarType, typename SimplexFadTypesT::EnergyFad   >::value ||
                                std::is_same< ScalarType, typename SimplexFadTypesT::MomentumFad >::value;
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

/***************************************************************************//**
 *  \brief Base class for automatic differentiation types used in fluid problems
 *  \tparam SpaceDim    (integer) spatial dimensions
 *  \tparam SimplexPhysicsT simplex fluid dynamic physics type
 ******************************************************************************/
template <typename SimplexPhysicsT>
struct EvaluationTypes
{
    static constexpr Plato::OrdinalType mNumSpatialDims        = SimplexPhysicsT::mNumSpatialDims;        /*!< number of spatial dimensions */
    static constexpr Plato::OrdinalType mNumNodesPerCell       = SimplexPhysicsT::mNumNodesPerCell;       /*!< number of nodes per simplex cell */
    static constexpr Plato::OrdinalType mNumControlDofsPerNode = SimplexPhysicsT::mNumControlDofsPerNode; /*!< number of design variable fields */
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
    using GradPrevMass     = GradPreviousMassTypes<SimplexPhysicsT>;

    using GradCurEnergy    = GradCurrentEnergyTypes<SimplexPhysicsT>;
    using GradPrevEnergy   = GradPreviousEnergyTypes<SimplexPhysicsT>;

    using GradCurMomentum  = GradCurrentMomentumTypes<SimplexPhysicsT>;
    using GradPrevMomentum = GradPreviousMomentumTypes<SimplexPhysicsT>;
    using GradPredictor    = GradMomentumPredictorTypes<SimplexPhysicsT>;
};


template<typename PhysicsT>
struct WorkSetBuilder
{
private:
    using SimplexPhysicsT = typename PhysicsT::SimplexT;

    using ConfigLocalOridnalMap   = Plato::NodeCoordinate<SimplexPhysicsT::mNumSpatialDims>;

    using MassLocalOridnalMap     = Plato::VectorEntryOrdinal<SimplexPhysicsT::mNumSpatialDims, SimplexPhysicsT::mNumMassDofsPerNode>;
    using EnergyLocalOridnalMap   = Plato::VectorEntryOrdinal<SimplexPhysicsT::mNumSpatialDims, SimplexPhysicsT::mNumEnergyDofsPerNode>;
    using MomentumLocalOridnalMap = Plato::VectorEntryOrdinal<SimplexPhysicsT::mNumSpatialDims, SimplexPhysicsT::mNumMomentumDofsPerNode>;
    using ControlLocalOridnalMap  = Plato::VectorEntryOrdinal<SimplexPhysicsT::mNumSpatialDims, SimplexPhysicsT::mNumControlDofsPerNode>;

    using ConfigFad   = typename Plato::FluidMechanics::SimplexFadTypes<SimplexPhysicsT>::ConfigFad;
    using ControlFad  = typename Plato::FluidMechanics::SimplexFadTypes<SimplexPhysicsT>::ControlFad;
    using MassFad     = typename Plato::FluidMechanics::SimplexFadTypes<SimplexPhysicsT>::MassFad;
    using EnergyFad   = typename Plato::FluidMechanics::SimplexFadTypes<SimplexPhysicsT>::EnergyFad;
    using MomentumFad = typename Plato::FluidMechanics::SimplexFadTypes<SimplexPhysicsT>::MomentumFad;

public:
    void buildMomentumWorkSet
    (const Plato::SpatialDomain               & aDomain,
     const MomentumLocalOridnalMap            & aMap,
     const Plato::ScalarVector                & aInput,
     Plato::ScalarMultiVectorT<Plato::Scalar> & aOutput)
    {
        Plato::workset_state_scalar_scalar<
            SimplexPhysicsT::mNumMomentumDofsPerNode,
            SimplexPhysicsT::mNumNodesPerCell>
        (aDomain, aMap, aInput, aOutput);
    }

    void buildMomentumWorkSet
    (const Plato::OrdinalType                 & aNumCells,
     const MomentumLocalOridnalMap            & aMap,
     const Plato::ScalarVector                & aInput,
     Plato::ScalarMultiVectorT<Plato::Scalar> & aOutput)
    {
        Plato::workset_state_scalar_scalar<
            SimplexPhysicsT::mNumMomentumDofsPerNode,
            SimplexPhysicsT::mNumNodesPerCell>
        (aNumCells, aMap, aInput, aOutput);
    }

    void buildMomentumWorkSet
    (const Plato::SpatialDomain             & aDomain,
     const MomentumLocalOridnalMap          & aMap,
     const Plato::ScalarVector              & aInput,
     Plato::ScalarMultiVectorT<MomentumFad> & aOutput)
    {
        Plato::workset_state_scalar_fad<
            SimplexPhysicsT::mNumMomentumDofsPerNode,
            SimplexPhysicsT::mNumNodesPerCell,
            MomentumFad>
        (aDomain, aMap, aInput, aOutput);
    }

    void buildMomentumWorkSet
    (const Plato::OrdinalType               & aNumCells,
     const MomentumLocalOridnalMap          & aMap,
     const Plato::ScalarVector              & aInput,
     Plato::ScalarMultiVectorT<MomentumFad> & aOutput)
    {
        Plato::workset_state_scalar_fad<
            SimplexPhysicsT::mNumMomentumDofsPerNode,
            SimplexPhysicsT::mNumNodesPerCell,
            MomentumFad>
        (aNumCells, aMap, aInput, aOutput);
    }

    void buildEnergyWorkSet
    (const Plato::SpatialDomain               & aDomain,
     const EnergyLocalOridnalMap              & aMap,
     const Plato::ScalarVector                & aInput,
     Plato::ScalarMultiVectorT<Plato::Scalar> & aOutput)
    {
        Plato::workset_state_scalar_scalar<
            SimplexPhysicsT::mNumEnergyDofsPerNode,
            SimplexPhysicsT::mNumNodesPerCell>
        (aDomain, aMap, aInput, aOutput);
    }

    void buildEnergyWorkSet
    (const Plato::OrdinalType                 & aNumCells,
     const EnergyLocalOridnalMap              & aMap,
     const Plato::ScalarVector                & aInput,
     Plato::ScalarMultiVectorT<Plato::Scalar> & aOutput)
    {
        Plato::workset_state_scalar_scalar<
            SimplexPhysicsT::mNumEnergyDofsPerNode,
            SimplexPhysicsT::mNumNodesPerCell>
        (aNumCells, aMap, aInput, aOutput);
    }

    void buildEnergyWorkSet
    (const Plato::SpatialDomain              & aDomain,
     const EnergyLocalOridnalMap             & aMap,
     const Plato::ScalarVector               & aInput,
     Plato::ScalarMultiVectorT<EnergyFad>    & aOutput)
    {
        Plato::workset_state_scalar_fad<
            SimplexPhysicsT::mNumEnergyDofsPerNode,
            SimplexPhysicsT::mNumNodesPerCell,
            EnergyFad>
        (aDomain, aMap, aInput, aOutput);
    }

    void buildEnergyWorkSet
    (const Plato::OrdinalType                & aNumCells,
     const EnergyLocalOridnalMap             & aMap,
     const Plato::ScalarVector               & aInput,
     Plato::ScalarMultiVectorT<EnergyFad>    & aOutput)
    {
        Plato::workset_state_scalar_fad<
            SimplexPhysicsT::mNumEnergyDofsPerNode,
            SimplexPhysicsT::mNumNodesPerCell,
            EnergyFad>
        (aNumCells, aMap, aInput, aOutput);
    }

    void buildMassWorkSet
    (const Plato::SpatialDomain               & aDomain,
     const MassLocalOridnalMap                & aMap,
     const Plato::ScalarVector                & aInput,
     Plato::ScalarMultiVectorT<Plato::Scalar> & aOutput)
    {
        Plato::workset_state_scalar_scalar<
            SimplexPhysicsT::mNumMassDofsPerNode,
            SimplexPhysicsT::mNumNodesPerCell>
        (aDomain, aMap, aInput, aOutput);
    }

    void buildMassWorkSet
    (const Plato::OrdinalType                 & aNumCells,
     const MassLocalOridnalMap                & aMap,
     const Plato::ScalarVector                & aInput,
     Plato::ScalarMultiVectorT<Plato::Scalar> & aOutput)
    {
        Plato::workset_state_scalar_scalar<
            SimplexPhysicsT::mNumMassDofsPerNode,
            SimplexPhysicsT::mNumNodesPerCell>
        (aNumCells, aMap, aInput, aOutput);
    }

    void buildMassWorkSet
    (const Plato::SpatialDomain         & aDomain,
     const MassLocalOridnalMap          & aMap,
     const Plato::ScalarVector          & aInput,
     Plato::ScalarMultiVectorT<MassFad> & aOutput)
    {
        Plato::workset_state_scalar_fad<
            SimplexPhysicsT::mNumMassDofsPerNode,
            SimplexPhysicsT::mNumNodesPerCell,
            MassFad>
        (aDomain, aMap, aInput, aOutput);
    }

    void buildMassWorkSet
    (const Plato::OrdinalType           & aNumCells,
     const MassLocalOridnalMap          & aMap,
     const Plato::ScalarVector          & aInput,
     Plato::ScalarMultiVectorT<MassFad> & aOutput)
    {
        Plato::workset_state_scalar_fad<
            SimplexPhysicsT::mNumMassDofsPerNode,
            SimplexPhysicsT::mNumNodesPerCell,
            MassFad>
        (aNumCells, aMap, aInput, aOutput);
    }

    void buildControlWorkSet
    (const Plato::SpatialDomain               & aDomain,
     const ControlLocalOridnalMap             & aMap,
     const Plato::ScalarVector                & aInput,
     Plato::ScalarMultiVectorT<Plato::Scalar> & aOutput)
    {
        Plato::workset_state_scalar_scalar<
            SimplexPhysicsT::mNumControlDofsPerNode,
            SimplexPhysicsT::mNumNodesPerCell>
        (aDomain, aMap, aInput, aOutput);
    }

    void buildControlWorkSet
    (const Plato::OrdinalType                 & aNumCells,
     const ControlLocalOridnalMap             & aMap,
     const Plato::ScalarVector                & aInput,
     Plato::ScalarMultiVectorT<Plato::Scalar> & aOutput)
    {
        Plato::workset_state_scalar_scalar<
            SimplexPhysicsT::mNumControlDofsPerNode,
            SimplexPhysicsT::mNumNodesPerCell>
        (aNumCells, aMap, aInput, aOutput);
    }

    void buildControlWorkSet
    (const Plato::SpatialDomain            & aDomain,
     const ControlLocalOridnalMap          & aMap,
     const Plato::ScalarVector             & aInput,
     Plato::ScalarMultiVectorT<ControlFad> & aOutput)
    {
        Plato::workset_state_scalar_fad<
            SimplexPhysicsT::mNumControlDofsPerNode,
            SimplexPhysicsT::mNumNodesPerCell,
            ControlFad>
        (aDomain, aMap, aInput, aOutput);
    }

    void buildControlWorkSet
    (const Plato::OrdinalType              & aNumCells,
     const ControlLocalOridnalMap          & aMap,
     const Plato::ScalarVector             & aInput,
     Plato::ScalarMultiVectorT<ControlFad> & aOutput)
    {
        Plato::workset_state_scalar_fad<
            SimplexPhysicsT::mNumControlDofsPerNode,
            SimplexPhysicsT::mNumNodesPerCell,
            ControlFad>
        (aNumCells, aMap, aInput, aOutput);
    }

    void buildConfigWorkSet
    (const Plato::SpatialDomain           & aDomain,
     const ConfigLocalOridnalMap          & aMap,
     Plato::ScalarArray3DT<Plato::Scalar> & aOutput)
    {
        Plato::workset_config_scalar<
            SimplexPhysicsT::mNumConfigDofsPerNode,
            SimplexPhysicsT::mNumNodesPerCell>
        (aDomain, aMap, aOutput);
    }

    void buildConfigWorkSet
    (const Plato::OrdinalType             & aNumCells,
     const ConfigLocalOridnalMap          & aMap,
     Plato::ScalarArray3DT<Plato::Scalar> & aOutput)
    {
        Plato::workset_config_scalar<
            SimplexPhysicsT::mNumConfigDofsPerNode,
            SimplexPhysicsT::mNumNodesPerCell>
        (aNumCells, aMap, aOutput);
    }

    void buildConfigWorkSet
    (const Plato::SpatialDomain       & aDomain,
     const ConfigLocalOridnalMap      & aMap,
     Plato::ScalarArray3DT<ConfigFad> & aOutput)
    {
        Plato::workset_config_fad<
            SimplexPhysicsT::mNumSpatialDims,
            SimplexPhysicsT::mNumNodesPerCell,
            SimplexPhysicsT::mNumConfigDofsPerNode,
            ConfigFad>
        (aDomain, aMap, aOutput);
    }

    void buildConfigWorkSet
    (const Plato::OrdinalType         & aNumCells,
     const ConfigLocalOridnalMap      & aMap,
     Plato::ScalarArray3DT<ConfigFad> & aOutput)
    {
        Plato::workset_config_fad<
            SimplexPhysicsT::mNumSpatialDims,
            SimplexPhysicsT::mNumNodesPerCell,
            SimplexPhysicsT::mNumConfigDofsPerNode,
            ConfigFad>
        (aNumCells, aMap, aOutput);
    }
};


template
<typename EvaluationT,
 typename PhysicsT>
inline void
build_scalar_function_worksets
(const Plato::SpatialDomain              & aDomain,
 const Plato::ScalarVector               & aControls,
 const Plato::Variables                  & aVariables,
 const Plato::LocalOrdinalMaps<PhysicsT> & aMaps,
       Plato::WorkSets                   & aWorkSets)
{
    auto tNumCells = aDomain.numCells();
    Plato::FluidMechanics::WorkSetBuilder<PhysicsT> tWorkSetBuilder;

    using CurrentVelocityT = typename EvaluationT::CurrentMomentumScalarType;
    auto tCurVelWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<CurrentVelocityT> > >
        ( Plato::ScalarMultiVectorT<CurrentVelocityT>("current velocity", tNumCells, PhysicsT::SimplexT::mNumMomentumDofsPerCell) );
    tWorkSetBuilder.buildMomentumWorkSet(aDomain, aMaps.mVectorStateOrdinalMap, aVariables.vector("current velocity"), tCurVelWS->mData);
    aWorkSets.set("current velocity", tCurVelWS);

    using CurrentPressureT = typename EvaluationT::CurrentMassScalarType;
    auto tCurPressWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<CurrentPressureT> > >
        ( Plato::ScalarMultiVectorT<CurrentPressureT>("current pressure", tNumCells, PhysicsT::SimplexT::mNumMassDofsPerCell) );
    tWorkSetBuilder.buildMassWorkSet(aDomain, aMaps.mScalarStateOrdinalMap, aVariables.vector("current pressure"), tCurPressWS->mData);
    aWorkSets.set("current pressure", tCurPressWS);

    using CurrentTemperatureT = typename EvaluationT::CurrentEnergyScalarType;
    auto tCurTempWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<CurrentTemperatureT> > >
        ( Plato::ScalarMultiVectorT<CurrentTemperatureT>("current temperature", tNumCells, PhysicsT::SimplexT::mNumEnergyDofsPerCell) );
    tWorkSetBuilder.buildEnergyWorkSet(aDomain, aMaps.mScalarStateOrdinalMap, aVariables.vector("current temperature"), tCurTempWS->mData);
    aWorkSets.set("current temperature", tCurTempWS);

    using ControlT = typename EvaluationT::ControlScalarType;
    auto tControlWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<ControlT> > >
        ( Plato::ScalarMultiVectorT<ControlT>("control", tNumCells, PhysicsT::SimplexT::mNumNodesPerCell) );
    tWorkSetBuilder.buildControlWorkSet(aDomain, aMaps.mControlOrdinalMap, aControls, tControlWS->mData);
    aWorkSets.set("control", tControlWS);

    auto tTimeStepsWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVector > >
        ( Plato::ScalarMultiVector("time steps", tNumCells, PhysicsT::SimplexT::mNumNodesPerCell) );
    Plato::workset_control_scalar_scalar<PhysicsT::SimplexT::mNumNodesPerCell>
        (aDomain, aMaps.mScalarStateOrdinalMap, aVariables.vector("time steps"), tTimeStepsWS->mData);
    aWorkSets.set("time steps", tTimeStepsWS);

    using ConfigT = typename EvaluationT::ConfigScalarType;
    auto tConfig = std::make_shared< Plato::MetaData< Plato::ScalarArray3DT<ConfigT> > >
        ( Plato::ScalarArray3DT<ConfigT>("configuration", tNumCells, PhysicsT::SimplexT::mNumNodesPerCell, PhysicsT::SimplexT::mNumConfigDofsPerNode) );
    tWorkSetBuilder.buildConfigWorkSet(aDomain, aMaps.mNodeCoordinate, tConfig->mData);
    aWorkSets.set("configuration", tConfig);
}

template
<typename EvaluationT,
 typename PhysicsT>
inline void
build_scalar_function_worksets
(const Plato::OrdinalType                & aNumCells,
 const Plato::ScalarVector               & aControls,
 const Plato::Variables                  & aVariables,
 const Plato::LocalOrdinalMaps<PhysicsT> & aMaps,
       Plato::WorkSets                   & aWorkSets)
{
    Plato::FluidMechanics::WorkSetBuilder<PhysicsT> tWorkSetBuilder;

    using CurrentVelocityT = typename EvaluationT::CurrentMomentumScalarType;
    auto tCurVelWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<CurrentVelocityT> > >
        ( Plato::ScalarMultiVectorT<CurrentVelocityT>("current velocity", aNumCells, PhysicsT::SimplexT::mNumMomentumDofsPerCell) );
    tWorkSetBuilder.buildMomentumWorkSet(aNumCells, aMaps.mVectorStateOrdinalMap, aVariables.vector("current velocity"), tCurVelWS->mData);
    aWorkSets.set("current velocity", tCurVelWS);

    using CurrentPressureT = typename EvaluationT::CurrentMassScalarType;
    auto tCurPressWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<CurrentPressureT> > >
        ( Plato::ScalarMultiVectorT<CurrentPressureT>("current pressure", aNumCells, PhysicsT::SimplexT::mNumMassDofsPerCell) );
    tWorkSetBuilder.buildMassWorkSet(aNumCells, aMaps.mScalarStateOrdinalMap, aVariables.vector("current pressure"), tCurPressWS->mData);
    aWorkSets.set("current pressure", tCurPressWS);

    using CurrentTemperatureT = typename EvaluationT::CurrentEnergyScalarType;
    auto tCurTempWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<CurrentTemperatureT> > >
        ( Plato::ScalarMultiVectorT<CurrentTemperatureT>("current temperature", aNumCells, PhysicsT::SimplexT::mNumEnergyDofsPerCell) );
    tWorkSetBuilder.buildEnergyWorkSet(aNumCells, aMaps.mScalarStateOrdinalMap, aVariables.vector("current temperature"), tCurTempWS->mData);
    aWorkSets.set("current temperature", tCurTempWS);

    using ControlT = typename EvaluationT::ControlScalarType;
    auto tControlWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<ControlT> > >
        ( Plato::ScalarMultiVectorT<ControlT>("control", aNumCells, PhysicsT::SimplexT::mNumNodesPerCell) );
    tWorkSetBuilder.buildControlWorkSet(aNumCells, aMaps.mControlOrdinalMap, aControls, tControlWS->mData);
    aWorkSets.set("control", tControlWS);

    auto tTimeStepsWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVector > >
        ( Plato::ScalarMultiVector("time steps", aNumCells, PhysicsT::SimplexT::mNumNodesPerCell) );
    Plato::workset_control_scalar_scalar<PhysicsT::SimplexT::mNumNodesPerCell>
        (aNumCells, aMaps.mScalarStateOrdinalMap, aVariables.vector("time steps"), tTimeStepsWS->mData);
    aWorkSets.set("time steps", tTimeStepsWS);

    using ConfigT = typename EvaluationT::ConfigScalarType;
    auto tConfig = std::make_shared< Plato::MetaData< Plato::ScalarArray3DT<ConfigT> > >
        ( Plato::ScalarArray3DT<ConfigT>("configuration", aNumCells, PhysicsT::SimplexT::mNumNodesPerCell, PhysicsT::SimplexT::mNumConfigDofsPerNode) );
    tWorkSetBuilder.buildConfigWorkSet(aNumCells, aMaps.mNodeCoordinate, tConfig->mData);
    aWorkSets.set("configuration", tConfig);
}

template
<typename EvaluationT,
 typename PhysicsT>
inline void
build_vector_function_worksets
(const Plato::SpatialDomain              & aDomain,
 const Plato::ScalarVector               & aControls,
 const Plato::Variables                  & aVariables,
 const Plato::LocalOrdinalMaps<PhysicsT> & aMaps,
       Plato::WorkSets                   & aWorkSets)

{
    auto tNumCells = aDomain.numCells();
    Plato::FluidMechanics::WorkSetBuilder<PhysicsT> tWorkSetBuilder;

    using CurrentPredictorT = typename EvaluationT::MomentumPredictorScalarType;
    auto tPredictorWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<CurrentPredictorT> > >
        ( Plato::ScalarMultiVectorT<CurrentPredictorT>("current predictor", tNumCells, PhysicsT::SimplexT::mNumMomentumDofsPerCell) );
    tWorkSetBuilder.buildMomentumWorkSet(aDomain, aMaps.mVectorStateOrdinalMap, aVariables.vector("current predictor"), tPredictorWS->mData);
    aWorkSets.set("current predictor", tPredictorWS);

    using CurrentVelocityT = typename EvaluationT::CurrentMomentumScalarType;
    auto tCurVelWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<CurrentVelocityT> > >
        ( Plato::ScalarMultiVectorT<CurrentVelocityT>("current velocity", tNumCells, PhysicsT::SimplexT::mNumMomentumDofsPerCell) );
    tWorkSetBuilder.buildMomentumWorkSet(aDomain, aMaps.mVectorStateOrdinalMap, aVariables.vector("current velocity"), tCurVelWS->mData);
    aWorkSets.set("current velocity", tCurVelWS);

    using CurrentPressureT = typename EvaluationT::CurrentMassScalarType;
    auto tCurPressWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<CurrentPressureT> > >
        ( Plato::ScalarMultiVectorT<CurrentPressureT>("current pressure", tNumCells, PhysicsT::SimplexT::mNumMassDofsPerCell) );
    tWorkSetBuilder.buildMassWorkSet(aDomain, aMaps.mScalarStateOrdinalMap, aVariables.vector("current pressure"), tCurPressWS->mData);
    aWorkSets.set("current pressure", tCurPressWS);

    using CurrentTemperatureT = typename EvaluationT::CurrentEnergyScalarType;
    auto tCurTempWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<CurrentTemperatureT> > >
        ( Plato::ScalarMultiVectorT<CurrentTemperatureT>("current temperature", tNumCells, PhysicsT::SimplexT::mNumEnergyDofsPerCell) );
    tWorkSetBuilder.buildEnergyWorkSet(aDomain, aMaps.mScalarStateOrdinalMap, aVariables.vector("current temperature"), tCurTempWS->mData);
    aWorkSets.set("current temperature", tCurTempWS);

    using PreviousVelocityT = typename EvaluationT::PreviousMomentumScalarType;
    auto tPrevVelWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<PreviousVelocityT> > >
        ( Plato::ScalarMultiVectorT<PreviousVelocityT>("previous velocity", tNumCells, PhysicsT::SimplexT::mNumMomentumDofsPerCell) );
    tWorkSetBuilder.buildMomentumWorkSet(aDomain, aMaps.mVectorStateOrdinalMap, aVariables.vector("previous velocity"), tPrevVelWS->mData);
    aWorkSets.set("previous velocity", tPrevVelWS);

    using PreviousPressureT = typename EvaluationT::PreviousMassScalarType;
    auto tPrevPressWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<PreviousPressureT> > >
        ( Plato::ScalarMultiVectorT<PreviousPressureT>("previous pressure", tNumCells, PhysicsT::SimplexT::mNumMassDofsPerCell) );
    tWorkSetBuilder.buildMassWorkSet(aDomain, aMaps.mScalarStateOrdinalMap, aVariables.vector("previous pressure"), tPrevPressWS->mData);
    aWorkSets.set("previous pressure", tPrevPressWS);

    using PreviousTemperatureT = typename EvaluationT::PreviousEnergyScalarType;
    auto tPrevTempWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<PreviousTemperatureT> > >
        ( Plato::ScalarMultiVectorT<PreviousTemperatureT>("previous temperature", tNumCells, PhysicsT::SimplexT::mNumEnergyDofsPerCell) );
    tWorkSetBuilder.buildEnergyWorkSet(aDomain, aMaps.mScalarStateOrdinalMap, aVariables.vector("previous temperature"), tPrevTempWS->mData);
    aWorkSets.set("previous temperature", tPrevTempWS);

    using ControlT = typename EvaluationT::ControlScalarType;
    auto tControlWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<ControlT> > >
        ( Plato::ScalarMultiVectorT<ControlT>("control", tNumCells, PhysicsT::SimplexT::mNumNodesPerCell) );
    tWorkSetBuilder.buildControlWorkSet(aDomain, aMaps.mControlOrdinalMap, aControls, tControlWS->mData);
    aWorkSets.set("control", tControlWS);

    using ConfigT = typename EvaluationT::ConfigScalarType;
    auto tConfig = std::make_shared< Plato::MetaData< Plato::ScalarArray3DT<ConfigT> > >
        ( Plato::ScalarArray3DT<ConfigT>("configuration", tNumCells, PhysicsT::SimplexT::mNumNodesPerCell, PhysicsT::SimplexT::mNumConfigDofsPerNode) );
    tWorkSetBuilder.buildConfigWorkSet(aDomain, aMaps.mNodeCoordinate, tConfig->mData);
    aWorkSets.set("configuration", tConfig);

    auto tTimeStepsWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVector > >
        ( Plato::ScalarMultiVector("time steps", tNumCells, PhysicsT::SimplexT::mNumNodesPerCell) );
    Plato::workset_control_scalar_scalar<PhysicsT::SimplexT::mNumNodesPerCell>
        (aDomain, aMaps.mScalarStateOrdinalMap, aVariables.vector("time steps"), tTimeStepsWS->mData);
    aWorkSets.set("time steps", tTimeStepsWS);

    auto tVelBCsWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVector > >
        ( Plato::ScalarMultiVector("previous velocity", tNumCells, PhysicsT::SimplexT::mNumMomentumDofsPerCell) );
    tWorkSetBuilder.buildMomentumWorkSet(aDomain, aMaps.mVectorStateOrdinalMap, aVariables.vector("prescribed velocity"), tVelBCsWS->mData);
    aWorkSets.set("prescribed velocity", tVelBCsWS);

    if(aVariables.defined("artificial compressibility"))
    {
        auto tArtificialCompressWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVector > >
            ( Plato::ScalarMultiVector("artificial compressibility", tNumCells, PhysicsT::SimplexT::mNumNodesPerCell) );
        Plato::workset_control_scalar_scalar<PhysicsT::SimplexT::mNumNodesPerCell>
            (aDomain, aMaps.mScalarStateOrdinalMap, aVariables.vector("artificial compressibility"), tArtificialCompressWS->mData);
        aWorkSets.set("artificial compressibility", tArtificialCompressWS);
    }
}

template
<typename EvaluationT,
 typename PhysicsT>
inline void
build_vector_function_worksets
(const Plato::OrdinalType                & aNumCells,
 const Plato::ScalarVector               & aControls,
 const Plato::Variables                  & aVariables,
 const Plato::LocalOrdinalMaps<PhysicsT> & aMaps,
       Plato::WorkSets                   & aWorkSets)
{
    Plato::FluidMechanics::WorkSetBuilder<PhysicsT> tWorkSetBuilder;

    using CurrentPredictorT = typename EvaluationT::MomentumPredictorScalarType;
    auto tPredictorWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<CurrentPredictorT> > >
        ( Plato::ScalarMultiVectorT<CurrentPredictorT>("current predictor", aNumCells, PhysicsT::SimplexT::mNumMomentumDofsPerCell) );
    tWorkSetBuilder.buildMomentumWorkSet(aNumCells, aMaps.mVectorStateOrdinalMap, aVariables.vector("current predictor"), tPredictorWS->mData);
    aWorkSets.set("current predictor", tPredictorWS);

    using CurrentVelocityT = typename EvaluationT::CurrentMomentumScalarType;
    auto tCurVelWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<CurrentVelocityT> > >
        ( Plato::ScalarMultiVectorT<CurrentVelocityT>("current velocity", aNumCells, PhysicsT::SimplexT::mNumMomentumDofsPerCell) );
    tWorkSetBuilder.buildMomentumWorkSet(aNumCells, aMaps.mVectorStateOrdinalMap, aVariables.vector("current velocity"), tCurVelWS->mData);
    aWorkSets.set("current velocity", tCurVelWS);

    using CurrentPressureT = typename EvaluationT::CurrentMassScalarType;
    auto tCurPressWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<CurrentPressureT> > >
        ( Plato::ScalarMultiVectorT<CurrentPressureT>("current pressure", aNumCells, PhysicsT::SimplexT::mNumMassDofsPerCell) );
    tWorkSetBuilder.buildMassWorkSet(aNumCells, aMaps.mScalarStateOrdinalMap, aVariables.vector("current pressure"), tCurPressWS->mData);
    aWorkSets.set("current pressure", tCurPressWS);

    using CurrentTemperatureT = typename EvaluationT::CurrentEnergyScalarType;
    auto tCurTempWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<CurrentTemperatureT> > >
        ( Plato::ScalarMultiVectorT<CurrentTemperatureT>("current temperature", aNumCells, PhysicsT::SimplexT::mNumEnergyDofsPerCell) );
    tWorkSetBuilder.buildEnergyWorkSet(aNumCells, aMaps.mScalarStateOrdinalMap, aVariables.vector("current temperature"), tCurTempWS->mData);
    aWorkSets.set("current temperature", tCurTempWS);

    using PreviousVelocityT = typename EvaluationT::PreviousMomentumScalarType;
    auto tPrevVelWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<PreviousVelocityT> > >
        ( Plato::ScalarMultiVectorT<PreviousVelocityT>("previous velocity", aNumCells, PhysicsT::SimplexT::mNumMomentumDofsPerCell) );
    tWorkSetBuilder.buildMomentumWorkSet(aNumCells, aMaps.mVectorStateOrdinalMap, aVariables.vector("previous velocity"), tPrevVelWS->mData);
    aWorkSets.set("previous velocity", tPrevVelWS);

    using PreviousPressureT = typename EvaluationT::PreviousMassScalarType;
    auto tPrevPressWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<PreviousPressureT> > >
        ( Plato::ScalarMultiVectorT<PreviousPressureT>("previous pressure", aNumCells, PhysicsT::SimplexT::mNumMassDofsPerCell) );
    tWorkSetBuilder.buildMassWorkSet(aNumCells, aMaps.mScalarStateOrdinalMap, aVariables.vector("previous pressure"), tPrevPressWS->mData);
    aWorkSets.set("previous pressure", tPrevPressWS);

    using PreviousTemperatureT = typename EvaluationT::PreviousEnergyScalarType;
    auto tPrevTempWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<PreviousTemperatureT> > >
        ( Plato::ScalarMultiVectorT<PreviousTemperatureT>("previous temperature", aNumCells, PhysicsT::SimplexT::mNumEnergyDofsPerCell) );
    tWorkSetBuilder.buildEnergyWorkSet(aNumCells, aMaps.mScalarStateOrdinalMap, aVariables.vector("previous temperature"), tPrevTempWS->mData);
    aWorkSets.set("previous temperature", tPrevTempWS);

    using ControlT = typename EvaluationT::ControlScalarType;
    auto tControlWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<ControlT> > >
        ( Plato::ScalarMultiVectorT<ControlT>("control", aNumCells, PhysicsT::SimplexT::mNumNodesPerCell) );
    tWorkSetBuilder.buildControlWorkSet(aNumCells, aMaps.mControlOrdinalMap, aControls, tControlWS->mData);
    aWorkSets.set("control", tControlWS);

    using ConfigT = typename EvaluationT::ConfigScalarType;
    auto tConfig = std::make_shared< Plato::MetaData< Plato::ScalarArray3DT<ConfigT> > >
        ( Plato::ScalarArray3DT<ConfigT>("configuration", aNumCells, PhysicsT::SimplexT::mNumNodesPerCell, PhysicsT::SimplexT::mNumConfigDofsPerNode) );
    tWorkSetBuilder.buildConfigWorkSet(aNumCells, aMaps.mNodeCoordinate, tConfig->mData);
    aWorkSets.set("configuration", tConfig);

    auto tTimeStepsWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVector > >
        ( Plato::ScalarMultiVector("time steps", aNumCells, PhysicsT::SimplexT::mNumNodesPerCell) );
    Plato::workset_control_scalar_scalar<PhysicsT::SimplexT::mNumNodesPerCell>
        (aNumCells, aMaps.mScalarStateOrdinalMap, aVariables.vector("time steps"), tTimeStepsWS->mData);
    aWorkSets.set("time steps", tTimeStepsWS);

    auto tVelBCsWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVector > >
        ( Plato::ScalarMultiVector("previous velocity", aNumCells, PhysicsT::SimplexT::mNumMomentumDofsPerCell) );
    tWorkSetBuilder.buildMomentumWorkSet(aNumCells, aMaps.mVectorStateOrdinalMap, aVariables.vector("prescribed velocity"), tVelBCsWS->mData);
    aWorkSets.set("prescribed velocity", tVelBCsWS);

    if(aVariables.defined("artificial compressibility"))
    {
        auto tArtificialCompressWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVector > >
            ( Plato::ScalarMultiVector("artificial compressibility", aNumCells, PhysicsT::SimplexT::mNumNodesPerCell) );
        Plato::workset_control_scalar_scalar<PhysicsT::SimplexT::mNumNodesPerCell>
            (aNumCells, aMaps.mScalarStateOrdinalMap, aVariables.vector("artificial compressibility"), tArtificialCompressWS->mData);
        aWorkSets.set("artificial compressibility", tArtificialCompressWS);
    }
}



// todo: abstract scalar function
template<typename PhysicsT, typename EvaluationT>
class AbstractScalarFunction
{
private:
    using ResultT = typename EvaluationT::ResultScalarType;

public:
    AbstractScalarFunction(){}
    virtual ~AbstractScalarFunction(){}

    virtual std::string name() const = 0;
    virtual void evaluate(const Plato::WorkSets & aWorkSets, Plato::ScalarVectorT<ResultT> & aResult) const = 0;
    virtual void evaluateBoundary(const Plato::WorkSets & aWorkSets, Plato::ScalarVectorT<ResultT> & aResult) const = 0;
};
// class AbstractScalarFunction

template<typename PhysicsT, typename EvaluationT>
class AverageSurfacePressure : public Plato::FluidMechanics::AbstractScalarFunction<PhysicsT, EvaluationT>
{
private:
    static constexpr auto mNumSpatialDims       = PhysicsT::SimplexT::mNumSpatialDims;         /*!< number of spatial dimensions */
    static constexpr auto mNumSpatialDimsOnFace = PhysicsT::SimplexT::mNumSpatialDimsOnFace;   /*!< number of spatial dimensions on face */
    static constexpr auto mNumNodesPerFace      = PhysicsT::SimplexT::mNumNodesPerFace;        /*!< number of nodes per face */
    static constexpr auto mNumPressDofsPerNode  = PhysicsT::SimplexT::mNumMassDofsPerNode;     /*!< number of pressure dofs per node */

    using ResultT   = typename EvaluationT::ResultScalarType;
    using ConfigT   = typename EvaluationT::ConfigScalarType;
    using PressureT = typename EvaluationT::CurrentMassScalarType;

    // set local typenames
    using CubatureRule  = Plato::LinearTetCubRuleDegreeOne<mNumSpatialDimsOnFace>;

    // member metadata
    Plato::DataMap& mDataMap;
    CubatureRule mCubatureRule;
    const Plato::SpatialDomain& mSpatialDomain;

    // member parameters
    std::string mFuncName;
    std::vector<std::string> mSideSets;

public:
    AverageSurfacePressure
    (const std::string          & aName,
     const Plato::SpatialDomain & aDomain,
     Plato::DataMap             & aDataMap,
     Teuchos::ParameterList     & aInputs) :
         mDataMap(aDataMap),
         mCubatureRule(CubatureRule()),
         mSpatialDomain(aDomain),
         mFuncName(aName)
    {
        auto tMyCriteria = aInputs.sublist("Criteria").sublist(aName);
        mSideSets = Plato::parse_array<std::string>("Sides", tMyCriteria);
    }

    virtual ~AverageSurfacePressure(){}

    std::string name() const override { return mFuncName; }

    void evaluate(const Plato::WorkSets & aWorkSets, Plato::ScalarVectorT<ResultT> & aResult) const override
    { return; }

    void evaluateBoundary(const Plato::WorkSets & aWorkSets, Plato::ScalarVectorT<ResultT> & aResult) const override
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
        auto tConfigurationWS   = Plato::metadata<Plato::ScalarArray3DT<ConfigT>>(aWorkSets.get("configuration"));
        auto tCurrentPressureWS = Plato::metadata<Plato::ScalarMultiVectorT<PressureT>>(aWorkSets.get("current pressure"));
        for(auto& tName : mSideSets)
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
                    Plato::OrdinalType tLocalNodeOrdinals[mNumSpatialDims];
                    auto tCellOrdinal = tFace2Elems_elems[tElem];
                    tCreateFaceLocalNode2ElemLocalNodeIndexMap(tCellOrdinal, tFaceOrdinal, tCell2Verts, tFace2Verts, tLocalNodeOrdinals);

                    // calculate surface Jacobian and surface integral weight
                    ConfigT tSurfaceAreaTimesCubWeight(0.0);
                    tComputeSurfaceJacobians(tCellOrdinal, aFaceI, tLocalNodeOrdinals, tConfigurationWS, tJacobians);
                    tComputeSurfaceIntegralWeight(aFaceI, tCubatureWeight, tJacobians, tSurfaceAreaTimesCubWeight);

                    // evaluate surface scalar function
                    tIntrplScalarField(tCellOrdinal, tBasisFunctions, tCurrentPressureWS, tCurrentPressGP);

                    // calculate surface integral, which is defined as
                    // \int_{\Gamma_e}N_p^a p^h d\Gamma_e
                    for( Plato::OrdinalType tNode=0; tNode < mNumNodesPerFace; tNode++)
                    {
                        for( Plato::OrdinalType tDof=0; tDof < mNumPressDofsPerNode; tDof++)
                        {
                            aResult(tCellOrdinal) += tBasisFunctions(tNode) *
                                tCurrentPressGP(tCellOrdinal) * tSurfaceAreaTimesCubWeight;
                        }
                    }
                }
            }, "average surface pressure integral");

        }
    }
};
// class AverageSurfacePressure

template<typename PhysicsT, typename EvaluationT>
class AverageSurfaceTemperature : public Plato::FluidMechanics::AbstractScalarFunction<PhysicsT, EvaluationT>
{
private:
    static constexpr auto mNumSpatialDims       = PhysicsT::SimplexT::mNumSpatialDims;         /*!< number of spatial dimensions */
    static constexpr auto mNumSpatialDimsOnFace = PhysicsT::SimplexT::mNumSpatialDimsOnFace;   /*!< number of spatial dimensions on face */
    static constexpr auto mNumNodesPerFace      = PhysicsT::SimplexT::mNumNodesPerFace;        /*!< number of nodes per face */
    static constexpr auto mNumPressDofsPerNode  = PhysicsT::SimplexT::mNumMassDofsPerNode;     /*!< number of temperature dofs per node */

    using TempT    = typename EvaluationT::CurrentEnergyScalarType;
    using ResultT  = typename EvaluationT::ResultScalarType;
    using ConfigT  = typename EvaluationT::ConfigScalarType;

    // set local typenames
    using CubatureRule  = Plato::LinearTetCubRuleDegreeOne<mNumSpatialDimsOnFace>;

    // member metadata
    Plato::DataMap& mDataMap;
    CubatureRule mCubatureRule;
    const Plato::SpatialDomain& mSpatialDomain;

    // member parameters
    std::string mFuncName;
    std::vector<std::string> mWallSets;

public:
    AverageSurfaceTemperature
    (const std::string          & aName,
     const Plato::SpatialDomain & aDomain,
     Plato::DataMap             & aDataMap,
     Teuchos::ParameterList     & aInputs) :
         mDataMap(aDataMap),
         mCubatureRule(CubatureRule()),
         mSpatialDomain(aDomain),
         mFuncName(aName)
    {
        auto tMyCriteria = aInputs.sublist("Criteria").sublist(aName);
        mWallSets = Plato::parse_array<std::string>("Sides", tMyCriteria);
    }

    virtual ~AverageSurfaceTemperature(){}

    std::string name() const override { return mFuncName; }

    void evaluate(const Plato::WorkSets & aWorkSets, Plato::ScalarVectorT<ResultT> & aResult) const override
    { return; }

    void evaluateBoundary(const Plato::WorkSets & aWorkSets, Plato::ScalarVectorT<ResultT> & aResult) const override
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
        Plato::ScalarVectorT<TempT> tCurrentTempGP("current temperature at Gauss point", tNumCells);

        // set input worksets
        auto tCurrentTempWS   = Plato::metadata<Plato::ScalarMultiVectorT<TempT>>(aWorkSets.get("current temperature"));
        auto tConfigurationWS = Plato::metadata<Plato::ScalarArray3DT<ConfigT>>(aWorkSets.get("configuration"));

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
                    Plato::OrdinalType tLocalNodeOrdinals[mNumSpatialDims];
                    auto tCellOrdinal = tFace2Elems_elems[tElem];
                    tCreateFaceLocalNode2ElemLocalNodeIndexMap(tCellOrdinal, tFaceOrdinal, tCell2Verts, tFace2Verts, tLocalNodeOrdinals);

                    // calculate surface Jacobian and surface integral weight
                    ConfigT tSurfaceAreaTimesCubWeight(0.0);
                    tComputeSurfaceJacobians(tCellOrdinal, aFaceI, tLocalNodeOrdinals, tConfigurationWS, tJacobians);
                    tComputeSurfaceIntegralWeight(aFaceI, tCubatureWeight, tJacobians, tSurfaceAreaTimesCubWeight);

                    // evaluate surface scalar function
                    tIntrplScalarField(tCellOrdinal, tBasisFunctions, tCurrentTempWS, tCurrentTempGP);

                    // calculate surface integral, which is defined as
                    // \int_{\Gamma_e}N_p^a p^h d\Gamma_e
                    for( Plato::OrdinalType tNode=0; tNode < mNumNodesPerFace; tNode++)
                    {
                        for( Plato::OrdinalType tDof=0; tDof < mNumPressDofsPerNode; tDof++)
                        {
                            aResult(tCellOrdinal) += tBasisFunctions(tNode) *
                                tCurrentTempGP(tCellOrdinal) * tSurfaceAreaTimesCubWeight;
                        }
                    }
                }
            }, "average surface temperature integral");

        }
    }
};
// class AverageSurfaceTemperature


template<Plato::OrdinalType NumNodesPerCell, typename ControlT>
DEVICE_TYPE inline ControlT
simp_penalization
(const Plato::OrdinalType & aCellOrdinal,
 const Plato::Scalar      & aPhysicalParam,
 const Plato::Scalar      & aPenaltyParam,
 const Plato::Scalar      & aMinErsatzParam,
 const Plato::ScalarMultiVectorT<ControlT> & aControlWS)
{
    ControlT tDensity = Plato::cell_density<NumNodesPerCell>(aCellOrdinal, aControlWS);
    ControlT tPenalizedDensity = aMinErsatzParam +
        ( (static_cast<Plato::Scalar>(1.0) - aMinErsatzParam) * pow(tDensity, aPenaltyParam) );
    ControlT tPenalizedParam = tPenalizedDensity * aPhysicalParam;
    return tPenalizedParam;
}

template<Plato::OrdinalType NumNodesPerCell, typename ControlT>
DEVICE_TYPE inline ControlT
ramp_penalization
(const Plato::OrdinalType & aCellOrdinal,
 const Plato::Scalar      & aPhysicalParam,
 const Plato::Scalar      & aConvexityParam,
 const Plato::ScalarMultiVectorT<ControlT> & aControlWS)
{
    ControlT tDensity = Plato::cell_density<NumNodesPerCell>(aCellOrdinal, aControlWS);
    ControlT tPenalizedPhysicalParam =
        ( tDensity * ( aPhysicalParam * (static_cast<Plato::Scalar>(1.0) - aConvexityParam)
            - static_cast<Plato::Scalar>(1.0) ) + static_cast<Plato::Scalar>(1.0) )
            / ( aPhysicalParam * (static_cast<Plato::Scalar>(1.0) + aConvexityParam * tDensity) );
    return tPenalizedPhysicalParam;
}

template<Plato::OrdinalType NumNodesPerCell, typename ControlT>
DEVICE_TYPE inline ControlT
brinkman_penalization
(const Plato::OrdinalType & aCellOrdinal,
 const Plato::Scalar      & aPhysicalParam,
 const Plato::Scalar      & aConvexityParam,
 const Plato::ScalarMultiVectorT<ControlT> & aControlWS)
{
    ControlT tDensity = Plato::cell_density<NumNodesPerCell>(aCellOrdinal, aControlWS);
    ControlT tPenalizedPhysicalParam = aPhysicalParam * (static_cast<Plato::Scalar>(1.0) - tDensity)
        / (static_cast<Plato::Scalar>(1.0) + (aConvexityParam * tDensity));
    return tPenalizedPhysicalParam;
}

template<Plato::OrdinalType NumNodesPerCell,
         Plato::OrdinalType NumSpaceDim,
         typename AViewTypeT,
         typename BViewTypeT,
         typename CViewTypeT>
DEVICE_TYPE inline void
strain_rate
(const Plato::OrdinalType & aCellOrdinal,
 const AViewTypeT & aStateWS,
 const BViewTypeT & aGradient,
 const CViewTypeT & aStrainRate)
{
    // calculate strain rate for incompressible flows, which is defined as
    // \frac{1}{2}\left( \frac{\partial u_i}{\partial x_j} + \frac{\partial u_j}{\partial x_i} \right)
    for(Plato::OrdinalType tNode = 0; tNode < NumNodesPerCell; tNode++)
    {
        for(Plato::OrdinalType tDimI = 0; tDimI < NumSpaceDim; tDimI++)
        {
            for(Plato::OrdinalType tDimJ = 0; tDimJ < NumSpaceDim; tDimJ++)
            {
                auto tLocalDimI = tNode * NumSpaceDim + tDimI;
                auto tLocalDimJ = tNode * NumSpaceDim + tDimJ;
                aStrainRate(aCellOrdinal, tDimI, tDimJ) += static_cast<Plato::Scalar>(0.5) *
                    ( ( aGradient(aCellOrdinal, tNode, tDimJ) * aStateWS(aCellOrdinal, tLocalDimI) )
                    + ( aGradient(aCellOrdinal, tNode, tDimI) * aStateWS(aCellOrdinal, tLocalDimJ) ) );
            }
        }
    }
}





// todo: internal energy
// calculate internal energy, which is defined as
//   \f$ \int_{\Omega_e}\left[ \tau_{ij}(\theta):\tau_{ij}(\theta) + \alpha(\theta)u_i^2 \right] d\Omega_e, \f$
// where \f$\theta\f$ denotes the controls, \f$\alpha\f$ denotes the Brinkman penalization parameter.
template<typename PhysicsT, typename EvaluationT>
class InternalDissipationEnergyIncompressible : public Plato::FluidMechanics::AbstractScalarFunction<PhysicsT, EvaluationT>
{
private:
    static constexpr auto mNumSpatialDims    = PhysicsT::SimplexT::mNumSpatialDims;         /*!< number of spatial dimensions */
    static constexpr auto mNumNodesPerCell   = PhysicsT::SimplexT::mNumNodesPerCell;        /*!< number of nodes per cell */
    static constexpr auto mNumVelDofsPerNode = PhysicsT::SimplexT::mNumMomentumDofsPerCell; /*!< number of velocity dofs per node */

    // local forward automatic differentiation typenames
    using ResultT  = typename EvaluationT::ResultScalarType;
    using CurVelT  = typename EvaluationT::CurrentMomentumScalarType;
    using ConfigT  = typename EvaluationT::ConfigScalarType;
    using ControlT = typename EvaluationT::ControlScalarType;
    using StrainT  = typename Plato::FluidMechanics::fad_type_t<typename PhysicsT::SimplexT, CurVelT, ConfigT>;

    // set local typenames
    using CubatureRule  = Plato::LinearTetCubRuleDegreeOne<mNumSpatialDims>;

    // member parameters
    std::string mFuncName;
    Plato::Scalar mPrNum = 1.0;
    Plato::Scalar mDaNum = 1.0;
    Plato::Scalar mPrNumConvexityParam = 0.5;
    Plato::Scalar mBrinkmanConvexityParam = 0.5;

    // member metadata
    Plato::DataMap& mDataMap;
    CubatureRule mCubatureRule;
    const Plato::SpatialDomain& mSpatialDomain;

public:
    InternalDissipationEnergyIncompressible
    (const std::string          & aName,
     const Plato::SpatialDomain & aDomain,
     Plato::DataMap             & aDataMap,
     Teuchos::ParameterList     & aInputs) :
         mFuncName(aName),
         mDataMap(aDataMap),
         mCubatureRule(CubatureRule()),
         mSpatialDomain(aDomain)
    {
        mDaNum = Plato::parse_parameter<Plato::Scalar>("Darcy Number", "Dimensionless Properties", aInputs);
        mPrNum = Plato::parse_parameter<Plato::Scalar>("Prandtl Number", "Dimensionless Properties", aInputs);
        this->setPenaltyModels(aInputs);
    }

    virtual ~InternalDissipationEnergyIncompressible(){}

    std::string name() const override { return mFuncName; }

    void evaluate(const Plato::WorkSets & aWorkSets, Plato::ScalarVectorT<ResultT> & aResult) const override
    {
        // set local functors
        Plato::ComputeGradientWorkset<mNumSpatialDims> tComputeGradient;
        Plato::InterpolateFromNodal<mNumSpatialDims, mNumVelDofsPerNode, 0/*offset*/, mNumSpatialDims> tIntrplVectorField;

        // set local worksets
        auto tNumCells = mSpatialDomain.numCells();
        Plato::ScalarVectorT<ConfigT> tVolumeTimesWeight("volume times gauss weight", tNumCells);
        Plato::ScalarVectorT<CurVelT> tCurVelDotCurVel("current velocity dot current velocity", tNumCells);
        Plato::ScalarArray3DT<ConfigT> tGradient("gradient", tNumCells, mNumNodesPerCell, mNumSpatialDims);
        Plato::ScalarArray3DT<StrainT> tStrainRate("strain rate", tNumCells, mNumSpatialDims, mNumSpatialDims);
        Plato::ScalarArray3DT<ResultT> tDevStress("deviatoric stress", tNumCells, mNumSpatialDims, mNumSpatialDims);
        Plato::ScalarMultiVectorT<CurVelT> tCurVelGP("current velocity at Gauss point", tNumCells, mNumSpatialDims);

        // set input worksets
        auto tControlWS = Plato::metadata<Plato::ScalarMultiVectorT<ControlT>>(aWorkSets.get("control"));
        auto tConfigWS  = Plato::metadata<Plato::ScalarArray3DT<ConfigT>>(aWorkSets.get("configuration"));
        auto tCurVelWS  = Plato::metadata<Plato::ScalarMultiVectorT<CurVelT>>(aWorkSets.get("current velocity"));

        // transfer member data to device
        auto tPrNum = mPrNum;
        auto tDaNum = mDaNum;
        auto tPrConvexParam = mPrNumConvexityParam;
        auto tBrinkConvexParam = mBrinkmanConvexityParam;

        auto tCubWeight = mCubatureRule.getCubWeight();
        auto tBasisFunctions = mCubatureRule.getBasisFunctions();
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
        {
            tComputeGradient(aCellOrdinal, tGradient, tConfigWS, tVolumeTimesWeight);
            tVolumeTimesWeight(aCellOrdinal) *= tCubWeight;

            // calculate deviatoric stress contribution to internal energy
            Plato::FluidMechanics::strain_rate<mNumNodesPerCell, mNumSpatialDims>(aCellOrdinal, tCurVelWS, tGradient, tStrainRate);
            ControlT tPenalizedPrNum =
                Plato::FluidMechanics::ramp_penalization<mNumNodesPerCell>(aCellOrdinal, tPrNum, tPrConvexParam, tControlWS);
            ControlT tTwoTimesPenalizedPrNum = static_cast<Plato::Scalar>(2.0) * tPenalizedPrNum;
            Plato::blas2::scale<mNumSpatialDims, mNumSpatialDims>(aCellOrdinal, tTwoTimesPenalizedPrNum, tStrainRate, tDevStress);
            Plato::blas2::dot<mNumSpatialDims, mNumSpatialDims>(aCellOrdinal, tDevStress, tDevStress, aResult);

            // calculate fictitious material model (i.e. brinkman model) contribution to internal energy
            auto tPermeability = tPrNum / tDaNum;
            ControlT tPenalizedPermeability =
                Plato::FluidMechanics::brinkman_penalization<mNumNodesPerCell>(aCellOrdinal, tPermeability, tBrinkConvexParam, tControlWS);
            tIntrplVectorField(aCellOrdinal, tBasisFunctions, tCurVelWS, tCurVelGP);
            Plato::blas1::dot<mNumSpatialDims>(aCellOrdinal, tCurVelGP, tCurVelGP, tCurVelDotCurVel);
            aResult(aCellOrdinal) += tPenalizedPermeability * tCurVelDotCurVel(aCellOrdinal);

            // apply gauss weight times volume multiplier
            aResult(aCellOrdinal) *= tVolumeTimesWeight(aCellOrdinal);

        }, "internal energy");
    }

    void evaluateBoundary(const Plato::WorkSets & aWorkSets, Plato::ScalarVectorT<ResultT> & aResult) const override
    { return; }

private:
    void setPenaltyModels(Teuchos::ParameterList & aInputs)
    {
        auto tMyCriterionInputs = aInputs.sublist("Criteria").sublist(mFuncName);
        if(tMyCriterionInputs.isSublist("Penalty Function"))
        {
            auto tPenaltyFuncInputs = tMyCriterionInputs.sublist("Penalty Function");
            mPrNumConvexityParam = tPenaltyFuncInputs.get<Plato::Scalar>("Prandtl Convexity Parameter", 0.5);
            mBrinkmanConvexityParam = tPenaltyFuncInputs.get<Plato::Scalar>("Brinkman Convexity Parameter", 0.5);
        }
    }
};
// class InternalDissipationEnergyIncompressible


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
public:
    virtual ~CriterionBase(){}

    /******************************************************************************//**
     * \brief Return function name
     * \return user defined function name
     **********************************************************************************/
    virtual std::string name() const = 0;

    virtual Plato::Scalar value
    (const Plato::ScalarVector & aControls,
     const Plato::Primal & aVariables) const = 0;

    virtual Plato::ScalarVector gradientConfig
    (const Plato::ScalarVector & aControls,
     const Plato::Primal & aVariables) const = 0;

    virtual Plato::ScalarVector gradientControl
    (const Plato::ScalarVector & aControls,
     const Plato::Primal & aVariables) const = 0;

    virtual Plato::ScalarVector gradientCurrentPress
    (const Plato::ScalarVector & aControls,
     const Plato::Primal & aVariables) const = 0;

    virtual Plato::ScalarVector gradientCurrentTemp
    (const Plato::ScalarVector & aControls,
     const Plato::Primal & aVariables) const = 0;

    virtual Plato::ScalarVector gradientCurrentVel
    (const Plato::ScalarVector & aControls,
     const Plato::Primal & aVariables) const = 0;
};
// class ScalarFunctionBase


// todo: physics scalar function
template<typename PhysicsT>
class PhysicsScalarFunction : public Plato::FluidMechanics::CriterionBase
{
private:
    std::string mFuncName;

    static constexpr auto mNumSpatialDims         = PhysicsT::SimplexT::mNumSpatialDims;         /*!< number of spatial dimensions */
    static constexpr auto mNumNodesPerCell        = PhysicsT::SimplexT::mNumNodesPerCell;        /*!< number of nodes per cell */
    static constexpr auto mNumMassDofsPerCell     = PhysicsT::SimplexT::mNumMassDofsPerCell;     /*!< number of mass dofs per cell */
    static constexpr auto mNumEnergyDofsPerCell   = PhysicsT::SimplexT::mNumEnergyDofsPerCell;   /*!< number of energy dofs per cell */
    static constexpr auto mNumMomentumDofsPerCell = PhysicsT::SimplexT::mNumMomentumDofsPerCell; /*!< number of momentum dofs per cell */
    static constexpr auto mNumMassDofsPerNode     = PhysicsT::SimplexT::mNumMassDofsPerNode;     /*!< number of mass dofs per node */
    static constexpr auto mNumEnergyDofsPerNode   = PhysicsT::SimplexT::mNumEnergyDofsPerNode;   /*!< number of energy dofs per node */
    static constexpr auto mNumMomentumDofsPerNode = PhysicsT::SimplexT::mNumMomentumDofsPerNode; /*!< number of momentum dofs per node */
    static constexpr auto mNumConfigDofsPerCell   = PhysicsT::SimplexT::mNumConfigDofsPerCell;   /*!< number of configuration degrees of freedom per cell */
    static constexpr auto mNumControlDofsPerNode  = PhysicsT::SimplexT::mNumControlDofsPerNode;  /*!< number of design variables per node */

    // forward automatic differentiation evaluation types
    using ResidualEvalT     = typename Plato::FluidMechanics::Evaluation<typename PhysicsT::SimplexT>::Residual;
    using GradConfigEvalT   = typename Plato::FluidMechanics::Evaluation<typename PhysicsT::SimplexT>::GradConfig;
    using GradControlEvalT  = typename Plato::FluidMechanics::Evaluation<typename PhysicsT::SimplexT>::GradControl;
    using GradCurVelEvalT   = typename Plato::FluidMechanics::Evaluation<typename PhysicsT::SimplexT>::GradCurMomentum;
    using GradCurTempEvalT  = typename Plato::FluidMechanics::Evaluation<typename PhysicsT::SimplexT>::GradCurEnergy;
    using GradCurPressEvalT = typename Plato::FluidMechanics::Evaluation<typename PhysicsT::SimplexT>::GradCurMass;

    // element scalar functions types
    using ResidualFunc     = std::shared_ptr<Plato::FluidMechanics::AbstractScalarFunction<PhysicsT, ResidualEvalT>>;
    using GradConfigFunc   = std::shared_ptr<Plato::FluidMechanics::AbstractScalarFunction<PhysicsT, GradConfigEvalT>>;
    using GradControlFunc  = std::shared_ptr<Plato::FluidMechanics::AbstractScalarFunction<PhysicsT, GradControlEvalT>>;
    using GradCurVelFunc   = std::shared_ptr<Plato::FluidMechanics::AbstractScalarFunction<PhysicsT, GradCurVelEvalT>>;
    using GradCurTempFunc  = std::shared_ptr<Plato::FluidMechanics::AbstractScalarFunction<PhysicsT, GradCurTempEvalT>>;
    using GradCurPressFunc = std::shared_ptr<Plato::FluidMechanics::AbstractScalarFunction<PhysicsT, GradCurPressEvalT>>;

    // element scalar functions per element block, i.e. domain
    std::unordered_map<std::string, ResidualFunc>     mResidualFuncs;
    std::unordered_map<std::string, GradConfigFunc>   mGradConfigFuncs;
    std::unordered_map<std::string, GradControlFunc>  mGradControlFuncs;
    std::unordered_map<std::string, GradCurVelFunc>   mGradCurrentVelocityFuncs;
    std::unordered_map<std::string, GradCurPressFunc> mGradCurrentPressureFuncs;
    std::unordered_map<std::string, GradCurTempFunc>  mGradCurrentTemperatureFuncs;

    Plato::DataMap& mDataMap;
    const Plato::SpatialModel& mSpatialModel;
    Plato::LocalOrdinalMaps<PhysicsT> mLocalOrdinalMaps;

public:
    PhysicsScalarFunction
    (Plato::SpatialModel    & aModel,
     Plato::DataMap         & aDataMap,
     Teuchos::ParameterList & aInputs,
     std::string            & aName):
        mFuncName(aName),
        mSpatialModel(aModel),
        mDataMap(aDataMap),
        mLocalOrdinalMaps(aModel.Mesh)
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
     const Plato::Primal & aVariables) const
    {
        using ResultScalarT = typename ResidualEvalT::ResultScalarType;
        ResultScalarT tReturnValue(0.0);

        // evaluate internal domain
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            Plato::WorkSets tInputWorkSets;
            auto tNumCells = tDomain.numCells();
            Plato::FluidMechanics::build_scalar_function_worksets<ResidualEvalT>
                (tDomain, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            auto tName = tDomain.getDomainName();
            Plato::ScalarVectorT<ResultScalarT> tResultWS("cells value", tNumCells);
            mResidualFuncs.at(tName)->evaluate(tInputWorkSets, tResultWS);

            tReturnValue += Plato::local_result_sum<Plato::Scalar>(tNumCells, tResultWS);
        }

        // evaluate boundary
        {
            Plato::WorkSets tInputWorkSets;
            auto tNumCells = mSpatialModel.Mesh.nelems();
            Plato::FluidMechanics::build_scalar_function_worksets<ResidualEvalT>
                (tNumCells, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            Plato::ScalarVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells);
            mResidualFuncs.begin()->second->evaluateBoundary(tInputWorkSets, tResultWS);

            tReturnValue += Plato::local_result_sum<Plato::Scalar>(tNumCells, tResultWS);
        }

        return tReturnValue;
    }

    Plato::ScalarVector gradientConfig
    (const Plato::ScalarVector & aControls,
     const Plato::Primal & aVariables) const
    {
        using ResultScalarT = typename GradConfigEvalT::ResultScalarType;
        const auto tNumNodes = mSpatialModel.Mesh.nverts();
        Plato::ScalarVector tGradient("gradient wrt configuration", mNumSpatialDims * tNumNodes);

        // evaluate internal domain
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            Plato::WorkSets tInputWorkSets;
            auto tNumCells = tDomain.numCells();
            Plato::FluidMechanics::build_scalar_function_worksets<GradConfigEvalT>
                (tDomain, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            auto tName = tDomain.getDomainName();
            Plato::ScalarVectorT<ResultScalarT> tResultWS("cells value", tNumCells);
            mGradConfigFuncs.at(tName)->evaluate(tInputWorkSets, tResultWS);

            Plato::assemble_vector_gradient_fad<mNumNodesPerCell, mNumSpatialDims>
                (tDomain, mLocalOrdinalMaps.mVectorStateOrdinalMap, tResultWS, tGradient);
        }

        // evaluate boundary
        {
            Plato::WorkSets tInputWorkSets;
            auto tNumCells = mSpatialModel.Mesh.nelems();
            Plato::FluidMechanics::build_scalar_function_worksets<GradConfigEvalT>
                (tNumCells, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            Plato::ScalarVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells);
            mGradConfigFuncs.begin()->second->evaluateBoundary(tInputWorkSets, tResultWS);

            Plato::assemble_vector_gradient_fad<mNumNodesPerCell, mNumSpatialDims>
                (tNumCells, mLocalOrdinalMaps.mVectorStateOrdinalMap, tResultWS, tGradient);
        }

        return tGradient;
    }

    Plato::ScalarVector gradientControl
    (const Plato::ScalarVector & aControls,
     const Plato::Primal & aVariables) const
    {
        using ResultScalarT = typename GradControlEvalT::ResultScalarType;
        const auto tNumNodes = mSpatialModel.Mesh.nverts();
        Plato::ScalarVector tGradient("gradient wrt control", mNumControlDofsPerNode * tNumNodes);

        // evaluate internal domain
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            Plato::WorkSets tInputWorkSets;
            auto tNumCells = tDomain.numCells();
            Plato::FluidMechanics::build_scalar_function_worksets<GradControlEvalT>
                (tDomain, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            auto tName = tDomain.getDomainName();
            Plato::ScalarVectorT<ResultScalarT> tResultWS("cells value", tNumCells);
            mGradControlFuncs.at(tName)->evaluate(tInputWorkSets, tResultWS);

            Plato::assemble_vector_gradient_fad<mNumNodesPerCell, mNumControlDofsPerNode>
                (tDomain, mLocalOrdinalMaps.mControlOrdinalMap, tResultWS, tGradient);
        }

        // evaluate boundary
        {
            Plato::WorkSets tInputWorkSets;
            auto tNumCells = mSpatialModel.Mesh.nelems();
            Plato::FluidMechanics::build_scalar_function_worksets<GradControlEvalT>
                (tNumCells, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            Plato::ScalarVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells);
            mGradControlFuncs.begin()->second->evaluateBoundary(tInputWorkSets, tResultWS);

            Plato::assemble_vector_gradient_fad<mNumNodesPerCell, mNumControlDofsPerNode>
                (tNumCells, mLocalOrdinalMaps.mControlOrdinalMap, tResultWS, tGradient);
        }

        return tGradient;
    }

    Plato::ScalarVector gradientCurrentPress
    (const Plato::ScalarVector & aControls,
     const Plato::Primal & aVariables) const
    {
        using ResultScalarT = typename GradCurPressEvalT::ResultScalarType;
        const auto tNumNodes = mSpatialModel.Mesh.nverts();
        Plato::ScalarVector tGradient("gradient wrt current pressure state", mNumMassDofsPerNode * tNumNodes);

        // evaluate internal domain
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            Plato::WorkSets tInputWorkSets;
            auto tNumCells = tDomain.numCells();
            Plato::FluidMechanics::build_scalar_function_worksets<GradCurPressEvalT>
                (tDomain, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            auto tName = tDomain.getDomainName();
            Plato::ScalarVectorT<ResultScalarT> tResultWS("cells value", tNumCells);
            mGradCurrentPressureFuncs.at(tName)->evaluate(tInputWorkSets, tResultWS);

            Plato::assemble_vector_gradient_fad<mNumNodesPerCell, mNumMassDofsPerNode>
                (tDomain, mLocalOrdinalMaps.mScalarStateOrdinalMap, tResultWS, tGradient);
        }

        // evaluate boundary
        {
            Plato::WorkSets tInputWorkSets;
            auto tNumCells = mSpatialModel.Mesh.nelems();
            Plato::FluidMechanics::build_scalar_function_worksets<GradCurPressEvalT>
                (tNumCells, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            Plato::ScalarVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells);
            mGradCurrentPressureFuncs.begin()->second->evaluateBoundary(tInputWorkSets, tResultWS);

            Plato::assemble_vector_gradient_fad<mNumNodesPerCell, mNumMassDofsPerNode>
                (tNumCells, mLocalOrdinalMaps.mScalarStateOrdinalMap, tResultWS, tGradient);
        }

        return tGradient;
    }

    Plato::ScalarVector gradientCurrentTemp
    (const Plato::ScalarVector & aControls,
     const Plato::Primal & aVariables) const
    {
        using ResultScalarT = typename GradCurTempEvalT::ResultScalarType;
        const auto tNumNodes = mSpatialModel.Mesh.nverts();
        Plato::ScalarVector tGradient("gradient wrt current temperature state", mNumEnergyDofsPerNode * tNumNodes);

        // evaluate internal domain
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            Plato::WorkSets tInputWorkSets;
            auto tNumCells = tDomain.numCells();
            Plato::FluidMechanics::build_scalar_function_worksets<GradCurTempEvalT>
                (tDomain, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            auto tName = tDomain.getDomainName();
            Plato::ScalarVectorT<ResultScalarT> tResultWS("cells value", tNumCells);
            mGradCurrentTemperatureFuncs.at(tName)->evaluate(tInputWorkSets, tResultWS);

            Plato::assemble_vector_gradient_fad<mNumNodesPerCell, mNumEnergyDofsPerNode>
                (tDomain, mLocalOrdinalMaps.mScalarStateOrdinalMap, tResultWS, tGradient);
        }

        // evaluate boundary
        {
            Plato::WorkSets tInputWorkSets;
            auto tNumCells = mSpatialModel.Mesh.nelems();
            Plato::FluidMechanics::build_scalar_function_worksets<GradCurTempEvalT>
                (tNumCells, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            Plato::ScalarVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells);
            mGradCurrentTemperatureFuncs.begin()->second->evaluateBoundary(tInputWorkSets, tResultWS);

            Plato::assemble_vector_gradient_fad<mNumNodesPerCell, mNumEnergyDofsPerNode>
                (tNumCells, mLocalOrdinalMaps.mScalarStateOrdinalMap, tResultWS, tGradient);
        }

        return tGradient;
    }

    Plato::ScalarVector gradientCurrentVel
    (const Plato::ScalarVector & aControls,
     const Plato::Primal & aVariables) const
    {
        using ResultScalarT = typename GradCurVelEvalT::ResultScalarType;
        const auto tNumNodes = mSpatialModel.Mesh.nverts();
        Plato::ScalarVector tGradient("gradient wrt current velocity state", mNumMomentumDofsPerNode * tNumNodes);

        // evaluate internal domain
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            Plato::WorkSets tInputWorkSets;
            auto tNumCells = tDomain.numCells();
            Plato::FluidMechanics::build_scalar_function_worksets<GradCurVelEvalT>
                (tDomain, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            auto tName = tDomain.getDomainName();
            Plato::ScalarVectorT<ResultScalarT> tResultWS("cells value", tNumCells);
            mGradCurrentVelocityFuncs.at(tName)->evaluate(tInputWorkSets, tResultWS);

            Plato::assemble_vector_gradient_fad<mNumNodesPerCell, mNumMomentumDofsPerNode>
                (tDomain, mLocalOrdinalMaps.mVectorStateOrdinalMap, tResultWS, tGradient);
        }

        // evaluate boundary
        {
            Plato::WorkSets tInputWorkSets;
            auto tNumCells = mSpatialModel.Mesh.nelems();
            Plato::FluidMechanics::build_scalar_function_worksets<GradCurVelEvalT>
                (tNumCells, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            Plato::ScalarVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells);
            mGradCurrentVelocityFuncs.begin()->second->evaluateBoundary(tInputWorkSets, tResultWS);

            Plato::assemble_vector_gradient_fad<mNumNodesPerCell, mNumMomentumDofsPerNode>
                (tNumCells, mLocalOrdinalMaps.mVectorStateOrdinalMap, tResultWS, tGradient);
        }

        return tGradient;
    }

private:
    void initialize(Teuchos::ParameterList & aInputs)
    {
        typename PhysicsT::FunctionFactory tScalarFuncFactory;
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tName = tDomain.getDomainName();

            mResidualFuncs[tName] =
                tScalarFuncFactory.template createScalarFunction<PhysicsT, ResidualEvalT>
                    (mFuncName, tDomain, mDataMap, aInputs);

            mGradConfigFuncs[tName] =
                tScalarFuncFactory.template createScalarFunction<PhysicsT, GradConfigEvalT>
                    (mFuncName, tDomain, mDataMap, aInputs);

            mGradControlFuncs[tName] =
                tScalarFuncFactory.template createScalarFunction<PhysicsT, GradControlEvalT>
                    (mFuncName, tDomain, mDataMap, aInputs);

            mGradCurrentPressureFuncs[tName] =
                tScalarFuncFactory.template createScalarFunction<PhysicsT, GradCurPressEvalT>
                    (mFuncName, tDomain, mDataMap, aInputs);

            mGradCurrentTemperatureFuncs[tName] =
                tScalarFuncFactory.template createScalarFunction<PhysicsT, GradCurTempEvalT>
                    (mFuncName, tDomain, mDataMap, aInputs);

            mGradCurrentVelocityFuncs[tName] =
                tScalarFuncFactory.template createScalarFunction<PhysicsT, GradCurVelEvalT>
                    (mFuncName, tDomain, mDataMap, aInputs);
        }
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

    // set local type names
    using CubatureRule = Plato::LinearTetCubRuleDegreeOne<mNumSpatialDimsOnFace>;

    const std::string mSideSetName = "not defined"; /*!< side set name */

    CubatureRule mCubatureRule;  /*!< integration rule */
    const Plato::SpatialDomain& mSpatialDomain; /*!< Plato spatial model */

public:
    PressureSurfaceForces
    (const Plato::SpatialDomain & aSpatialDomain,
     std::string aSideSetName = "not defined") :
         mSideSetName(aSideSetName),
         mCubatureRule(CubatureRule()),
         mSpatialDomain(aSpatialDomain)
    {
        auto tLowerTag = Plato::tolower(mSideSetName);
        if(tLowerTag == "not defined")
        {
            THROWERR("Side set with name '" + mSideSetName + "' is not defined.")
        }
    }

    void operator()
    (const Plato::WorkSets & aWorkSets,
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
        auto tConfigWS    = Plato::metadata<Plato::ScalarArray3DT<ConfigT>>(aWorkSets.get("configuration"));
        auto tPrevPressWS = Plato::metadata<Plato::ScalarMultiVectorT<PrevPressT>>(aWorkSets.get("previous pressure"));

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



// todo DeviatoricSurfaceForces
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
    using StrainT  = typename Plato::FluidMechanics::fad_type_t<typename PhysicsT::SimplexT, PrevVelT, ConfigT>;

    // set local typenames
    using CubatureRule  = Plato::LinearTetCubRuleDegreeOne<mNumSpatialDimsOnFace>;

    std::string   mSideSetName; /*!< side set name */
    Plato::Scalar mPrNum = 1.0;
    Plato::Scalar mPrNumConvexityParam = 0.5;

    CubatureRule mCubatureRule;  /*!< integration rule */
    Omega_h::LOs mFaceOrdinalsOnBoundary;
    const Plato::SpatialDomain& mSpatialDomain; /*!< Plato spatial model */

public:
    DeviatoricSurfaceForces
    (const Plato::SpatialDomain & aSpatialDomain,
     Teuchos::ParameterList & aInputs,
     std::string aSideSetName = "") :
         mSideSetName(aSideSetName),
         mCubatureRule(CubatureRule()),
         mSpatialDomain(aSpatialDomain)
    {
        this->setParameters(aInputs);
        this->setFacesOnNonPrescribedBoundary(aInputs);
    }

    void operator()
    (const Plato::WorkSets & aWorkSets,
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
        Plato::ComputeGradientWorkset<mNumSpatialDims> tComputeGradient;
        Plato::NodeCoordinate<mNumSpatialDims> tCoords(&(mSpatialDomain.Mesh));
        Plato::ComputeSurfaceJacobians<mNumSpatialDims> tComputeSurfaceJacobians;
        Plato::ComputeSurfaceIntegralWeight<mNumSpatialDims> tComputeSurfaceIntegralWeight;
        Plato::CreateFaceLocalNode2ElemLocalNodeIndexMap<mNumSpatialDims> tCreateFaceLocalNode2ElemLocalNodeIndexMap;

        // transfer member data to device
        auto tPrNum = mPrNum;
        auto tPrNumConvexityParam = mPrNumConvexityParam;
        auto tFaceOrdinalsOnBoundary = mFaceOrdinalsOnBoundary;

        // set local data structures
        auto tNumCells = mSpatialDomain.Mesh.nelems();
        Plato::ScalarVectorT<ConfigT>  tCellVolume("cell weight", tNumCells);
        Plato::ScalarArray3DT<ConfigT> tGradient("cell gradient", tNumCells, mNumNodesPerCell, mNumSpatialDims);
        Plato::ScalarArray3DT<StrainT> tStrainRate("cell strain rate", tNumCells, mNumSpatialDims, mNumSpatialDims);
        auto tNumFaces = tFaceOrdinalsOnBoundary.size();
        Plato::ScalarArray3DT<ConfigT> tJacobians("cell jacobians", tNumFaces, mNumSpatialDimsOnFace, mNumSpatialDims);

        // set input state worksets
        auto tControlWS = Plato::metadata<Plato::ScalarMultiVectorT<ControlT>>(aWorkSets.get("control"));
        auto tConfigWS  = Plato::metadata<Plato::ScalarArray3DT<ConfigT>>(aWorkSets.get("configuration"));
        auto tPrevVelWS = Plato::metadata<Plato::ScalarMultiVectorT<PrevVelT>>(aWorkSets.get("previous velocity"));

        // calculate surface integral
        auto tCubatureWeight = mCubatureRule.getCubWeight();
        auto tBasisFunctions = mCubatureRule.getBasisFunctions();
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumFaces), LAMBDA_EXPRESSION(const Plato::OrdinalType & aFaceI)
        {
            auto tFaceOrdinal = tFaceOrdinalsOnBoundary[aFaceI];
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
                Plato::FluidMechanics::strain_rate<mNumNodesPerCell, mNumSpatialDims>
                    (tCellOrdinal, tPrevVelWS, tGradient, tStrainRate);

                // calculate penalized prandtl number
                ControlT tPenalizedPrandtlNum =
                    Plato::FluidMechanics::ramp_penalization<mNumNodesPerCell>(tCellOrdinal, tPrNum, tPrNumConvexityParam, tControlWS);

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
    void setParameters
    (Teuchos::ParameterList & aInputs)
    {
        if(aInputs.isSublist("Hyperbolic") == false)
        {
            THROWERR("'Hyperbolic' Parameter List is not defined.")
        }
        auto tHyperParamList = aInputs.sublist("Hyperbolic");

        mPrNum = Plato::parse_parameter<Plato::Scalar>("Prandtl Number", "Dimensionless Properties", tHyperParamList);
        this->setPenaltyModel(tHyperParamList);
    }

    void setPenaltyModel
    (Teuchos::ParameterList & aInputs)
    {
        if(aInputs.isSublist("Momentum Conservation") == false)
        {
            THROWERR(std::string("Parameter List 'Momentum Conservation' is not defined within Parameter List '")
                + aInputs.name() + "'.")
        }
        auto tMomentumParamList = aInputs.sublist("Momentum Conservation");

        if (tMomentumParamList.isSublist("Penalty Function"))
        {
            auto tPenaltyFuncList = tMomentumParamList.sublist("Penalty Function");
            mPrNumConvexityParam = tPenaltyFuncList.get<Plato::Scalar>("Prandtl Convexity Parameter", 0.5);
        }
    }

    void setFacesOnNonPrescribedBoundary(Teuchos::ParameterList& aInputs)
    {
        if (mSideSetName.empty())
        {
            if (aInputs.isSublist("Momentum Natural Boundary Conditions"))
            {
                auto tNaturalBCs = aInputs.sublist("Momentum Natural Boundary Conditions");
                auto tPrescribedSideSetNames = Plato::sideset_names(tNaturalBCs);
                mFaceOrdinalsOnBoundary =
                    Plato::entities_on_non_prescribed_boundary<Omega_h::EDGE, Omega_h::SIDE_SET>
                        (tPrescribedSideSetNames, mSpatialDomain.Mesh, mSpatialDomain.MeshSets);
            }
            else
            {
                // integral is performed along all the fluid surfaces
                std::vector<std::string> tEmptyPrescribedBCs;
                mFaceOrdinalsOnBoundary =
                    Plato::entities_on_non_prescribed_boundary<Omega_h::EDGE, Omega_h::SIDE_SET>
                        (tEmptyPrescribedBCs, mSpatialDomain.Mesh, mSpatialDomain.MeshSets);
            }
        }
        else
        {
            mFaceOrdinalsOnBoundary = Plato::side_set_face_ordinals(mSpatialDomain.MeshSets, mSideSetName);
            Plato::omega_h::print(mFaceOrdinalsOnBoundary, "non-prescribed boundary ordinals");
        }
    }
};
// class DeviatoricSurfaceForces

// todo: abstract vector function
template<typename PhysicsT, typename EvaluationT>
class AbstractVectorFunction
{
private:
    using ResultT = typename EvaluationT::ResultScalarType;

public:
    AbstractVectorFunction(){}
    virtual ~AbstractVectorFunction(){}

    virtual void evaluate
    (const Plato::WorkSets & aWorkSets,
     Plato::ScalarMultiVectorT<ResultT> & aResult) const = 0;

    virtual void evaluateBoundary
    (const Plato::SpatialModel & aSpatialModel,
     const Plato::WorkSets & aWorkSets,
     Plato::ScalarMultiVectorT<ResultT> & aResult) const = 0;

    virtual void evaluatePrescribed
    (const Plato::SpatialModel & aSpatialModel,
     const Plato::WorkSets & aWorkSets,
     Plato::ScalarMultiVectorT<ResultT> & aResult) const = 0;
};
// class AbstractVectorFunction







template
<Plato::OrdinalType NumNodesPerCell,
 Plato::OrdinalType NumSpatialDims,
 typename ResultT,
 typename ConfigT,
 typename PrevVelT>
DEVICE_TYPE inline void
calculate_convective_forces
(const Plato::OrdinalType & aCellOrdinal,
 const Plato::ScalarVector & aBasisFunctions,
 const Plato::ScalarVectorT<ConfigT> & aCellVolume,
 const Plato::ScalarArray3DT<ConfigT> & aGradient,
 const Plato::ScalarMultiVectorT<PrevVelT> & aPrevVelGP,
 const Plato::ScalarMultiVectorT<ResultT> & aResult)
{
    for(Plato::OrdinalType tNode = 0; tNode < NumNodesPerCell; tNode++)
    {
        for(Plato::OrdinalType tDimI = 0; tDimI < NumSpatialDims; tDimI++)
        {
            auto tDofIndex = (NumSpatialDims * tNode) + tDimI;
            for(Plato::OrdinalType tDimJ = 0; tDimJ < NumSpatialDims; tDimJ++)
            {
                aResult(aCellOrdinal, tDofIndex) +=
                    aCellVolume(aCellOrdinal) * aBasisFunctions(tNode) * ( aGradient(aCellOrdinal, tNode, tDimJ)
                        *  ( aPrevVelGP(aCellOrdinal, tDimJ) * aPrevVelGP(aCellOrdinal, tDimI) ) );
            }
        }
    }
}

template
<Plato::OrdinalType NumNodesPerCell,
 Plato::OrdinalType NumSpatialDims,
 typename ResultT,
 typename ConfigT,
 typename PrevVelT>
DEVICE_TYPE inline void
calculate_stabilized_convective_forces
(const Plato::OrdinalType & aCellOrdinal,
 const Plato::ScalarArray3DT<ConfigT> & aGradient,
 const Plato::ScalarMultiVectorT<PrevVelT> & aPrevVelGP,
 const Plato::ScalarMultiVectorT<ResultT> & aResult)
{
    for(Plato::OrdinalType tNode = 0; tNode < NumNodesPerCell; tNode++)
    {
        for(Plato::OrdinalType tDimI = 0; tDimI < NumSpatialDims; tDimI++)
        {
            for(Plato::OrdinalType tDimJ = 0; tDimJ < NumSpatialDims; tDimJ++)
            {
                aResult(aCellOrdinal, tDimI) += aGradient(aCellOrdinal, tNode, tDimJ) *
                    ( aPrevVelGP(aCellOrdinal, tDimJ) * aPrevVelGP(aCellOrdinal, tDimI) );
            }
        }
    }
}

template
<Plato::OrdinalType NumNodesPerCell,
 Plato::OrdinalType NumSpatialDims,
 typename ResultT,
 typename ConfigT,
 typename ControlT,
 typename StrainT>
DEVICE_TYPE inline void
calculate_vicous_forces
(const Plato::OrdinalType & aCellOrdinal,
 const ControlT & aPenalizedPrandtlNum,
 const Plato::ScalarVectorT<ConfigT> & aCellVolume,
 const Plato::ScalarArray3DT<ConfigT> & aGradient,
 const Plato::ScalarArray3DT<StrainT> & aStrainRate,
 const Plato::ScalarMultiVectorT<ResultT> & aResult)
{
    for(Plato::OrdinalType tNode = 0; tNode < NumNodesPerCell; tNode++)
    {
        for(Plato::OrdinalType tDimI = 0; tDimI < NumSpatialDims; tDimI++)
        {
            auto tDofIndex = (NumSpatialDims * tNode) + tDimI;
            for(Plato::OrdinalType tDimJ = 0; tDimJ < NumSpatialDims; tDimJ++)
            {
                aResult(aCellOrdinal, tDofIndex) += aCellVolume(aCellOrdinal) * aGradient(aCellOrdinal, tNode, tDimJ)
                    * ( static_cast<Plato::Scalar>(2.0) * aPenalizedPrandtlNum * aStrainRate(aCellOrdinal, tDimI, tDimJ) );
            }
        }
    }
}

template<typename PhysicsT, typename EvaluationT>
class VelocityPredictorResidual : public Plato::FluidMechanics::AbstractVectorFunction<PhysicsT, EvaluationT>
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
    static constexpr auto mNumConfigDofsPerCell = PhysicsT::SimplexT::mNumConfigDofsPerCell;   /*!< number of configuration degrees of freedom per cell */

    // set local ad types
    using ResultT    = typename EvaluationT::ResultScalarType;
    using ConfigT    = typename EvaluationT::ConfigScalarType;
    using ControlT   = typename EvaluationT::ControlScalarType;
    using PrevVelT   = typename EvaluationT::PreviousMomentumScalarType;
    using PrevTempT  = typename EvaluationT::PreviousEnergyScalarType;
    using PredictorT = typename EvaluationT::MomentumPredictorScalarType;
    using StrainT    = typename Plato::FluidMechanics::fad_type_t<typename PhysicsT::SimplexT, PrevVelT, ConfigT>;


    Plato::DataMap& mDataMap; /*!< output database */
    const Plato::SpatialDomain& mSpatialDomain; /*!< Plato spatial model */
    Plato::LinearTetCubRuleDegreeOne<mNumSpatialDims> mCubatureRule; /*!< cubature rule evaluator */

    // set local type names
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
    Plato::Scalar mGrNumMinErsatzMaterial = 1e-9;

    Plato::ScalarVector mGrNum;

public:
    VelocityPredictorResidual
    (const Plato::SpatialDomain & aDomain,
     Plato::DataMap             & aDataMap,
     Teuchos::ParameterList     & aInputs) :
         mDataMap(aDataMap),
         mSpatialDomain(aDomain),
         mCubatureRule(Plato::LinearTetCubRuleDegreeOne<mNumSpatialDims>()),
         mGrNum("grashof number", mNumSpatialDims)
    {
        this->setParameters(aInputs);
        this->setNaturalBoundaryConditions(aInputs);
        this->setStabilizedNaturalBoundaryConditions(aInputs);
    }

    virtual ~VelocityPredictorResidual(){}

    void evaluate
    (const Plato::WorkSets & aWorkSets,
     Plato::ScalarMultiVectorT<ResultT> & aResult)
    const override
    {
        auto tNumCells = mSpatialDomain.numCells();
        if( tNumCells != static_cast<Plato::OrdinalType>(aResult.extent(0)) )
        {
            THROWERR(std::string("Number of elements mismatch. Spatial domain and output/result workset ")
                + "have different number of cells. " + "Spatial domain has '" + std::to_string(tNumCells)
                + "' elements and output workset has '" + std::to_string(aResult.extent(0)) + "' elements.")
        }

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
        auto tTimeStepWS  = Plato::metadata<Plato::ScalarMultiVector>(aWorkSets.get("time steps"));
        auto tControlWS   = Plato::metadata<Plato::ScalarMultiVectorT<ControlT>>(aWorkSets.get("control"));
        auto tConfigWS    = Plato::metadata<Plato::ScalarArray3DT<ConfigT>>(aWorkSets.get("configuration"));
        auto tPrevVelWS   = Plato::metadata<Plato::ScalarMultiVectorT<PrevVelT>>(aWorkSets.get("previous velocity"));
        auto tPrevTempWS  = Plato::metadata<Plato::ScalarMultiVectorT<PrevTempT>>(aWorkSets.get("previous temperature"));
        auto tPredictorWS = Plato::metadata<Plato::ScalarMultiVectorT<PredictorT>>(aWorkSets.get("current predictor"));

        // transfer member data to device
        auto tDaNum = mDaNum;
        auto tPrNum = mPrNum;
        auto tGrNum = mGrNum;
        auto tPowerPenaltySIMP = mGrNumExponent;
        auto tPrNumConvexityParam = mPrNumConvexityParam;
        auto tBrinkmanConvexityParam = mBrinkmanConvexityParam;
        auto tMinErsatzMatSIMP = mGrNumMinErsatzMaterial;

        auto tCubWeight = mCubatureRule.getCubWeight();
        auto tBasisFunctions = mCubatureRule.getBasisFunctions();
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
        {
            tComputeGradient(aCellOrdinal, tGradient, tConfigWS, tCellVolume);
            tCellVolume(aCellOrdinal) *= tCubWeight;

            // calculate convective force integral, which are defined as
            // \int_{\Omega_e} N_u^a \left( \frac{\partial}{\partial x_j}(u^{n-1}_j u^{n-1}_i) \right) d\Omega_e
            tIntrplVectorField(aCellOrdinal, tBasisFunctions, tPrevVelWS, tPrevVelGP);
            Plato::FluidMechanics::calculate_convective_forces<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tBasisFunctions, tCellVolume, tGradient, tPrevVelGP, aResult);
            
            // calculate stabilized convective force integral, which are defined as
            // \frac{\partial}{\partial x_j}\left(u^{n-1}_j u^{n-1}_i\right)
            Plato::FluidMechanics::calculate_stabilized_convective_forces<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tGradient, tPrevVelGP, tStabForce);

            // calculate strain rate for incompressible flows, which is defined as
            // \frac{1}{2}\left( \frac{\partial u_i}{\partial x_j} + \frac{\partial u_j}{\partial x_i} \right)
            Plato::FluidMechanics::strain_rate<mNumNodesPerCell, mNumSpatialDims>(aCellOrdinal, tPrevVelWS, tGradient, tStrainRate);

            // calculate penalized prandtl number
            ControlT tPenalizedPrandtlNum =
                Plato::FluidMechanics::ramp_penalization<mNumNodesPerCell>(aCellOrdinal, tPrNum, tPrNumConvexityParam, tControlWS);

            // calculate viscous force integral, which are defined as,
            // \int_{\Omega_e}\frac{\partial N_u^a}{\partial x_j}\tau^h_{ij} d\Omega_e
            Plato::FluidMechanics::calculate_vicous_forces<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tPenalizedPrandtlNum, tCellVolume, tGradient, tStrainRate, aResult);

            // calculate natural convective force integral, which are defined as
            // \int_{\Omega_e} N_u^a \left(Gr_i Pr^2 T^h \right) d\Omega_e,
            // where e_i is the unit vector in the gravitational direction
            auto tPrNumTimesPrNum = tPrNum * tPrNum;
            ControlT tPenalizedPrNumSquared = Plato::FluidMechanics::simp_penalization<mNumNodesPerCell>
                (aCellOrdinal, tPrNumTimesPrNum, tPowerPenaltySIMP, tMinErsatzMatSIMP, tControlWS);
            tIntrplScalarField(aCellOrdinal, tBasisFunctions, tPrevTempWS, tPrevTempGP);
            /*ControlT tDensity = Plato::cell_density<mNumNodesPerCell>(aCellOrdinal, tControlWS);
            ControlT tPenalizedDensity = tMinErsatzMatSIMP +
                ( (static_cast<Plato::Scalar>(1.0) - tMinErsatzMatSIMP) * pow(tDensity, tPowerPenaltySIMP) );
            ControlT tPenalizedPrNumSquared = tPenalizedDensity * tPrNum * tPrNum;
            */
            for(Plato::OrdinalType tNode = 0; tNode < mNumNodesPerCell; tNode++)
            {
                for(Plato::OrdinalType tDim = 0; tDim < mNumSpatialDims; tDim++)
                {
                    auto tDofIndex = (mNumSpatialDims * tNode) + tDim;
                    aResult(aCellOrdinal, tDofIndex) += tCellVolume(aCellOrdinal) * tBasisFunctions(tNode)
                        * ( tGrNum(tDim) * tPenalizedPrNumSquared * tPrevTempGP(aCellOrdinal) );
                    tStabForce(aCellOrdinal, tDim) += tGrNum(tDim) * tPenalizedPrNumSquared * tPrevTempGP(aCellOrdinal);
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
   (const Plato::SpatialModel & aSpatialModel,
    const Plato::WorkSets & aWorkSets,
    Plato::ScalarMultiVectorT<ResultT> & aResult)
   const override
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
       auto tTimeStepWS = Plato::metadata<Plato::ScalarMultiVector>(aWorkSets.get("time steps"));
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
   (const Plato::SpatialModel & aSpatialModel,
    const Plato::WorkSets & aWorkSets,
    Plato::ScalarMultiVectorT<ResultT> & aResult)
   const override
   {
       if( mPrescribedBCs != nullptr )
       {
           // set input worksets
           auto tControlWS = Plato::metadata<Plato::ScalarMultiVectorT<ControlT>>(aWorkSets.get("control"));
           auto tConfigWS  = Plato::metadata<Plato::ScalarArray3DT<ConfigT>>(aWorkSets.get("configuration"));
           auto tPrevVelWS = Plato::metadata<Plato::ScalarMultiVectorT<PrevVelT>>(aWorkSets.get("previous velocity"));

           // calculate deviatoric traction forces, which are defined as
           // \int_{\Gamma_t} N_u^a\left(t_i + p^{n-1}n_i\right) d\Gamma
           auto tNumCells = aResult.extent(0);
           Plato::ScalarMultiVectorT<ResultT> tResultWS("traction forces", tNumCells, mNumDofsPerCell);
           mPrescribedBCs->get( aSpatialModel, tPrevVelWS, tControlWS, tConfigWS, tResultWS); // traction forces
           for(auto& tPair : mPressureBCs)
           {
               tPair.second->operator()(aWorkSets, tResultWS); // pressure forces on traction side sets
           }

           // multiply force vector by the corresponding nodal time steps
           auto tTimeStepWS = Plato::metadata<Plato::ScalarMultiVector>(aWorkSets.get("time steps"));
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
   void setParameters
   (Teuchos::ParameterList & aInputs)
   {
       if(aInputs.isSublist("Hyperbolic") == false)
       {
           THROWERR("'Hyperbolic' Parameter List is not defined.")
       }
       auto tHyperParamList = aInputs.sublist("Hyperbolic");
       this->setDimensionlessProperties(tHyperParamList);

       if(tHyperParamList.isSublist("Momentum Conservation") == false)
       {
           THROWERR(std::string("Parameter List 'Momentum Conservation' is not defined within Parameter List '") 
               + tHyperParamList.name() + "'.")
       }
       auto tMomentumParamList = tHyperParamList.sublist("Momentum Conservation");

       this->setPenaltyModel(tMomentumParamList);
   }

   void setPenaltyModel
   (Teuchos::ParameterList & aInputs)
   {
        if (aInputs.isSublist("Penalty Function"))
        {
            auto tPenaltyFuncList = aInputs.sublist("Penalty Function");
            mGrNumExponent = tPenaltyFuncList.get<Plato::Scalar>("Grashof Penalty Exponent", 3.0);
            mPrNumConvexityParam = tPenaltyFuncList.get<Plato::Scalar>("Prandtl Convexity Parameter", 0.5);
            mBrinkmanConvexityParam = tPenaltyFuncList.get<Plato::Scalar>("Brinkman Convexity Parameter", 0.5);
            mGrNumMinErsatzMaterial = tPenaltyFuncList.get<Plato::Scalar>("Grashof Minimum Ersatz Material Value", 0.0);
        }
   }

    void setDimensionlessProperties
    (Teuchos::ParameterList & aInputs)
    {
        mDaNum = Plato::parse_parameter<Plato::Scalar>("Darcy Number", "Dimensionless Properties", aInputs);
        mPrNum = Plato::parse_parameter<Plato::Scalar>("Prandtl Number", "Dimensionless Properties", aInputs);

        auto tGrNum = Plato::parse_parameter<Teuchos::Array<Plato::Scalar>>("Grashof Number", "Dimensionless Properties", aInputs);
        if(tGrNum.size() != mNumSpatialDims)
        {
            THROWERR(std::string("'Grashof Number' array length should match the number of physical spatial dimensions. ") 
                + "Array length is '" + std::to_string(tGrNum.size()) + "' and the number of physical spatial dimensions is '" 
                + std::to_string(mNumSpatialDims)  + "'.")
        }

        auto tLength = mGrNum.size();
        auto tHostGrNum = Kokkos::create_mirror(mGrNum);
        for(decltype(tLength) tDim = 0; tDim < tLength; tDim++)
        {
            tHostGrNum(tDim) = tGrNum[tDim];
        }
        Kokkos::deep_copy(mGrNum, tHostGrNum);
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
                    THROWERR(std::string("Error parsing Parameter List '") + tInputsNaturalBCs.name() 
                        + "'. Parameter List block is not valid. Expects Parameter Lists only.")
                }

                const std::string &tName = tInputsNaturalBCs.name(tItr);
                if(tInputsNaturalBCs.isSublist(tName) == false)
                {
                    THROWERR(std::string("Error parsing Parameter List '") + tInputsNaturalBCs.name() 
                        + "'. Parameter Sublist '" + tName.c_str() + "' is not defined.")
                }

                Teuchos::ParameterList &tSubList = tInputsNaturalBCs.sublist(tName);
                if(tSubList.isParameter("Sides") == false)
                {
                    THROWERR(std::string("Error parsing Parameter List '") + tSubList.name() + "'. 'Sides' keyword " 
                        +"is not define in Parameter Sublist '" + tName.c_str() + "'. The 'Sides' keyword is used " 
                        + "to define the 'side set' names where 'Natrual Boundary Conditions' are applied.")
                }
                const auto tSideSetName = tSubList.get<std::string>("Sides");

                mPressureBCs[tSideSetName] = std::make_shared<PressureForces>(mSpatialDomain, tSideSetName);
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
                    THROWERR(std::string("Error parsing Parameter List '") + tInputsNaturalBCs.name()
                        + "'. Parameter List block is not valid. Expects Parameter Lists only.")
                }

                const std::string &tName = tInputsNaturalBCs.name(tItr);
                if(tInputsNaturalBCs.isSublist(tName) == false)
                {
                    THROWERR(std::string("Error parsing Parameter List '") + tInputsNaturalBCs.name()
                        + "'. Parameter Sublist '" + tName.c_str() + "' is not defined.")
                }

                Teuchos::ParameterList &tSubList = tInputsNaturalBCs.sublist(tName);
                if(tSubList.isParameter("Sides") == false)
                {
                    THROWERR(std::string("Error parsing Parameter List '") + tSubList.name() + "'. 'Sides' keyword "
                        +"is not define in Parameter Sublist '" + tName.c_str() + "'. The 'Sides' keyword is used "
                        + "to define the 'side set' names where 'Natural Boundary Conditions' are applied.")
                }
                const auto tSideSetName = tSubList.get<std::string>("Sides");

                mDeviatoricBCs[tSideSetName] = std::make_shared<DeviatoricForces>(mSpatialDomain, aInputs, tSideSetName);
            }
        }
        else
        {
            mDeviatoricBCs["automated"] = std::make_shared<DeviatoricForces>(mSpatialDomain, aInputs);
        }
    }
};
// class VelocityPredictorResidual

template<typename PhysicsT, typename EvaluationT>
class VelocityIncrementResidual : public Plato::FluidMechanics::AbstractVectorFunction<PhysicsT, EvaluationT>
{
private:
    static constexpr auto mNumSpatialDims       = PhysicsT::SimplexT::mNumSpatialDims;         /*!< number of spatial dimensions */
    static constexpr auto mNumNodesPerCell      = PhysicsT::SimplexT::mNumNodesPerCell;        /*!< number of nodes per cell */
    static constexpr auto mNumVelDofsPerCell    = PhysicsT::SimplexT::mNumMomentumDofsPerCell; /*!< number of momentum dofs per cell */
    static constexpr auto mNumPressDofsPerCell  = PhysicsT::SimplexT::mNumMassDofsPerCell;     /*!< number of mass dofs per cell */
    static constexpr auto mNumVelDofsPerNode    = PhysicsT::SimplexT::mNumMomentumDofsPerNode; /*!< number of momentum dofs per node */
    static constexpr auto mNumPressDofsPerNode  = PhysicsT::SimplexT::mNumMassDofsPerNode;     /*!< number of mass dofs per node */
    static constexpr auto mNumConfigDofsPerCell = PhysicsT::SimplexT::mNumConfigDofsPerCell;   /*!< number of configuration degrees of freedom per cell */

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
    (const Plato::WorkSets & aWorkSets,
     Plato::ScalarMultiVectorT<ResultT> & aResult)
    const override
    {
        auto tNumCells = mSpatialDomain.numCells();
        if( tNumCells != static_cast<Plato::OrdinalType>(aResult.extent(0)) )
        {
            THROWERR(std::string("Number of elements mismatch. Spatial domain and output/result workset ")
                + "have different number of cells. " + "Spatial domain has '" + std::to_string(tNumCells)
                + "' elements and output workset has '" + std::to_string(aResult.extent(0)) + "' elements.")
        }

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
        auto tTimeStepWS  = Plato::metadata<Plato::ScalarMultiVector>(aWorkSets.get("time steps"));
        auto tConfigWS    = Plato::metadata<Plato::ScalarArray3DT<ConfigT>>(aWorkSets.get("configuration"));
        auto tCurVelWS    = Plato::metadata<Plato::ScalarMultiVectorT<CurVelT>>(aWorkSets.get("current velocity"));
        auto tPrevVelWS   = Plato::metadata<Plato::ScalarMultiVectorT<PrevVelT>>(aWorkSets.get("previous velocity"));
        auto tCurPressWS  = Plato::metadata<Plato::ScalarMultiVectorT<CurPressT>>(aWorkSets.get("current pressure"));
        auto tPrevPressWS = Plato::metadata<Plato::ScalarMultiVectorT<PrevPressT>>(aWorkSets.get("previous pressure"));
        auto tPredictorWS = Plato::metadata<Plato::ScalarMultiVectorT<PredictorT>>(aWorkSets.get("current predictor"));

        // transfer member data to device
        auto tThetaTwo = mThetaTwo;

        auto tCubWeight = mCubatureRule.getCubWeight();
        auto tBasisFunctions = mCubatureRule.getBasisFunctions();
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
    (const Plato::SpatialModel & aSpatialModel,
     const Plato::WorkSets & aWorkSets,
     Plato::ScalarMultiVectorT<ResultT> & aResult)
    const override
    { return; /* boundary integral equates zero */ }

    void evaluatePrescribed
    (const Plato::SpatialModel & aSpatialModel,
     const Plato::WorkSets & aWorkSets,
     Plato::ScalarMultiVectorT<ResultT> & aResult)
    const override
    { return; /* prescribed force integral equates zero */ }
};
// class VelocityIncrementResidual

template<typename PhysicsT, typename EvaluationT>
class TemperatureIncrementResidual : public Plato::FluidMechanics::AbstractVectorFunction<PhysicsT, EvaluationT>
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
    static constexpr auto mNumConfigDofsPerCell = PhysicsT::SimplexT::mNumConfigDofsPerCell;   /*!< number of configuration degrees of freedom per cell */

    // set local ad type
    using ResultT   = typename EvaluationT::ResultScalarType;
    using ConfigT   = typename EvaluationT::ConfigScalarType;
    using ControlT  = typename EvaluationT::ControlScalarType;
    using CurTempT  = typename EvaluationT::CurrentEnergyScalarType;
    using PrevVelT  = typename EvaluationT::PreviousMomentumScalarType;
    using PrevTempT = typename EvaluationT::PreviousEnergyScalarType;

    using StabForceT = typename Plato::FluidMechanics::fad_type_t<typename PhysicsT::SimplexT, PrevVelT, ConfigT, PrevTempT>;

    Plato::DataMap& mDataMap;                   /*!< output database */
    const Plato::SpatialDomain& mSpatialDomain; /*!< Plato spatial model */

    Plato::LinearTetCubRuleDegreeOne<mNumSpatialDims> mCubatureRule; /*!< integration rule */
    std::shared_ptr<Plato::NaturalBCs<mNumSpatialDims, mNumDofsPerNode>> mHeatFlux; /*!< heat flux evaluator */

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
        // todo add parsing for penalty and input parameters
        this->setSourceTerm(aInputs);
        this->setNaturalBoundaryConditions(aInputs);
    }

    virtual ~TemperatureIncrementResidual(){}

    void evaluate
    (const Plato::WorkSets & aWorkSets,
     Plato::ScalarMultiVectorT<ResultT> & aResult)
    const override
    {
        auto tNumCells = mSpatialDomain.numCells();
        if( tNumCells != static_cast<Plato::OrdinalType>(aResult.extent(0)) )
        {
            THROWERR(std::string("Number of elements mismatch. Spatial domain and output/result workset ")
                + "have different number of cells. " + "Spatial domain has '" + std::to_string(tNumCells)
                + "' elements and output workset has '" + std::to_string(aResult.extent(0)) + "' elements.")
        }

        // set local data
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
        auto tTimeStepWS = Plato::metadata<Plato::ScalarMultiVector>(aWorkSets.get("time steps"));
        auto tControlWS  = Plato::metadata<Plato::ScalarMultiVectorT<ControlT>>(aWorkSets.get("control"));
        auto tConfigWS   = Plato::metadata<Plato::ScalarArray3DT<ConfigT>>(aWorkSets.get("configuration"));
        auto tPrevVelWS  = Plato::metadata<Plato::ScalarMultiVectorT<PrevVelT>>(aWorkSets.get("previous velocity"));
        auto tCurTempWS  = Plato::metadata<Plato::ScalarMultiVectorT<CurTempT>>(aWorkSets.get("current temperature"));
        auto tPrevTempWS = Plato::metadata<Plato::ScalarMultiVectorT<PrevTempT>>(aWorkSets.get("previous temperature"));

        // transfer member data to device
        auto tHeatSource                = mHeatSource;
        auto tFluidDomainCutOff         = mFluidDomainCutOff;
        auto tCharacteristicLength      = mCharacteristicLength;
        auto tFluidThermalDiffusivity   = mFluidThermalDiffusivity;
        auto tSolidThermalDiffusivity   = mSolidThermalDiffusivity;
        auto tReferenceTempDifference   = mReferenceTempDifference;
        auto tFluidThermalConductivity  = mFluidThermalConductivity;
        auto tDiffusivityConvexityParam = mDiffusivityConvexityParam;

        auto tCubWeight = mCubatureRule.getCubWeight();
        auto tBasisFunctions = mCubatureRule.getBasisFunctions();
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
    (const Plato::SpatialModel & aSpatialModel,
     const Plato::WorkSets & aWorkSets,
     Plato::ScalarMultiVectorT<ResultT> & aResult)
    const override
    { return; /* boundary integral equates zero */ }

    void evaluatePrescribed
    (const Plato::SpatialModel & aSpatialModel,
     const Plato::WorkSets & aWorkSets,
     Plato::ScalarMultiVectorT<ResultT> & aResult)
    const override
    {
        if( mHeatFlux != nullptr )
        {
            // set input state worksets
            auto tConfigWS   = Plato::metadata<Plato::ScalarArray3DT<ConfigT>>(aWorkSets.get("configuration"));
            auto tControlWS  = Plato::metadata<Plato::ScalarMultiVectorT<ControlT>>(aWorkSets.get("control"));
            auto tPrevTempWS = Plato::metadata<Plato::ScalarMultiVectorT<PrevTempT>>(aWorkSets.get("previous temperature"));

            // evaluate prescribed flux
            auto tNumCells = aResult.extent(0);
            Plato::ScalarMultiVectorT<ResultT> tResultWS("heat flux", tNumCells, mNumDofsPerCell);
            mHeatFlux->get( aSpatialModel, tPrevTempWS, tControlWS, tConfigWS, tResultWS, -1.0 );

            auto tTimeStepWS = Plato::metadata<Plato::ScalarMultiVector>(aWorkSets.get("time steps"));
            Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
            {
                for(Plato::OrdinalType tDof = 0; tDof < mNumDofsPerCell; tDof++)
                {
                    aResult(aCellOrdinal, tDof) += tTimeStepWS(aCellOrdinal, tDof) * tResultWS(aCellOrdinal, tDof);
                }
            }, "heat flux contribution");
        }
    }

private:
    void setSourceTerm
    (Teuchos::ParameterList & aInputs)
    {
        if(aInputs.isSublist("Thermal Source Term"))
        {
            auto tSublist = aInputs.sublist("Thermal Source Term");
            mHeatSource = tSublist.get<Plato::Scalar>("Source Term", 0.0);
        }
    }

    void setNaturalBoundaryConditions
    (Teuchos::ParameterList & aInputs)
    {
        if(aInputs.isSublist("Thermal Natural Boundary Conditions"))
        {
            auto tSublist = aInputs.sublist("Thermal Natural Boundary Conditions");
            mHeatFlux = std::make_shared<Plato::NaturalBCs<mNumSpatialDims, mNumDofsPerNode>>(tSublist);
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

    const std::string mEntityName; /*!< side set name */
    const Plato::SpatialDomain& mSpatialDomain; /*!< Plato spatial model */
    Plato::LinearTetCubRuleDegreeOne<mNumSpatialDims> mCubatureRule;  /*!< integration rule */

public:
    MomentumSurfaceForces
    (const Plato::SpatialDomain & aSpatialDomain,
     const std::string & aEntityName) :
         mEntityName(aEntityName),
         mSpatialDomain(aSpatialDomain)
    {
    }

    void operator()
    (const Plato::WorkSets & aWorkSets,
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
        Plato::InterpolateFromNodal<mNumSpatialDims, mNumDofsPerNode> tIntrplVectorField;
        Plato::ComputeSurfaceIntegralWeight<mNumSpatialDims> tComputeSurfaceIntegralWeight;
        Plato::CreateFaceLocalNode2ElemLocalNodeIndexMap<mNumSpatialDims> tCreateFaceLocalNode2ElemLocalNodeIndexMap;

        // get sideset faces
        auto tFaceLocalOrdinals = Plato::side_set_face_ordinals(mSpatialDomain.MeshSets, mEntityName);
        auto tNumFaces = tFaceLocalOrdinals.size();
        Plato::ScalarArray3DT<ConfigT> tJacobians("jacobian", tNumFaces, mNumSpatialDimsOnFace, mNumSpatialDims);

        // set previous pressure at Gauss points container
        auto tNumCells = mSpatialDomain.Mesh.nelems();
        Plato::ScalarVector tVelBCsGP("prescribed velocities at Gauss point", tNumCells, mNumDofsPerNode);
        Plato::ScalarVectorT<PrevVelT> tPrevVelGP("previous velocity at Gauss point", tNumCells, mNumDofsPerNode);

        // set input state worksets
        auto tVelBCsWS  = Plato::metadata<Plato::ScalarMultiVector>(aWorkSets.get("prescribed velocity"));
        auto tConfigWS  = Plato::metadata<Plato::ScalarArray3DT<ConfigT>>(aWorkSets.get("configuration"));
        auto tPrevVelWS = Plato::metadata<Plato::ScalarMultiVectorT<PrevVelT>>(aWorkSets.get("previous velocity"));

        // evaluate integral
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
              tIntrplVectorField(tCellOrdinal, tBasisFunctions, tVelBCsWS, tVelBCsGP);
              tIntrplVectorField(tCellOrdinal, tBasisFunctions, tPrevVelWS, tPrevVelGP);
              for( Plato::OrdinalType tNode = 0; tNode < mNumNodesPerFace; tNode++ )
              {
                  for( Plato::OrdinalType tDof = 0; tDof < mNumDofsPerNode; tDof++ )
                  {
                      auto tCellDofOrdinal = (tLocalNodeOrd[tNode] * mNumDofsPerNode) + tDof;
                      aResult(tCellOrdinal, tCellDofOrdinal) += aMultiplier * tBasisFunctions(tNode) *
                          tUnitNormalVec(tDof) * ( tVelBCsGP(tDof) - tPrevVelGP(tDof) )* tWeight;
                  }
              }
          }
        }, "calculate surface momentum integral");
    }
};
// class MomentumSurfaceForces



template<typename PhysicsT, typename EvaluationT>
class PressureIncrementResidual : public Plato::FluidMechanics::AbstractVectorFunction<PhysicsT, EvaluationT>
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
    static constexpr auto mNumConfigDofsPerCell = PhysicsT::SimplexT::mNumConfigDofsPerCell;   /*!< number of configuration degrees of freedom per cell */

    using ResultT    = typename EvaluationT::ResultScalarType;
    using ConfigT    = typename EvaluationT::ConfigScalarType;
    using ControlT   = typename EvaluationT::ControlScalarType;
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
    std::unordered_map<std::string, std::shared_ptr<MomentumForces>> mMomentumBCs;

public:
    PressureIncrementResidual
    (const Plato::SpatialDomain & aDomain,
     Plato::DataMap             & aDataMap,
     Teuchos::ParameterList     & aInputs) :
         mDataMap(aDataMap),
         mSpatialDomain(aDomain),
         mCubatureRule(Plato::LinearTetCubRuleDegreeOne<mNumSpatialDims>())
    {
        this->setParameters(aInputs);
        this->setNaturalBoundaryConditions(aInputs);
    }

    virtual ~PressureIncrementResidual(){}

    void evaluate
    (const Plato::WorkSets & aWorkSets,
     Plato::ScalarMultiVectorT<ResultT> & aResult)
    const override
    {
        auto tNumCells = mSpatialDomain.numCells();
        if( tNumCells != static_cast<Plato::OrdinalType>(aResult.extent(0)) )
        {
            THROWERR(std::string("Number of elements mismatch. Spatial domain and output/result workset ")
                + "have different number of cells. " + "Spatial domain has '" + std::to_string(tNumCells)
                + "' elements and output workset has '" + std::to_string(aResult.extent(0)) + "' elements.")
        }

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
        auto tTimeStepWS  = Plato::metadata<Plato::ScalarMultiVector>(aWorkSets.get("time steps"));
        auto tACompressWS = Plato::metadata<Plato::ScalarMultiVector>(aWorkSets.get("artificial compressibility"));
        auto tConfigWS    = Plato::metadata<Plato::ScalarArray3DT<ConfigT>>(aWorkSets.get("configuration"));
        auto tPrevVelWS   = Plato::metadata<Plato::ScalarMultiVectorT<PrevVelT>>(aWorkSets.get("previous velocity"));
        auto tCurPressWS  = Plato::metadata<Plato::ScalarMultiVectorT<CurPressT>>(aWorkSets.get("current pressure"));
        auto tPrevPressWS = Plato::metadata<Plato::ScalarMultiVectorT<PrevPressT>>(aWorkSets.get("previous pressure"));
        auto tPredictorWS = Plato::metadata<Plato::ScalarMultiVectorT<PredictorT>>(aWorkSets.get("current predictor"));

        // transfer member data to device
        auto tThetaOne = mThetaOne;
        auto tThetaTwo = mThetaTwo;

        auto tCubWeight = mCubatureRule.getCubWeight();
        auto tBasisFunctions = mCubatureRule.getBasisFunctions();
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

    // todo: verify implementation with formulation - checked!
    void evaluateBoundary
    (const Plato::SpatialModel & aSpatialModel,
     const Plato::WorkSets & aWorkSets,
     Plato::ScalarMultiVectorT<ResultT> & aResult)
    const override
    {
        // calculate previous momentum forces, which are defined as
        // -\theta_1\Delta{t} \int_{\Gamma_u} N_u^a n_i( \hat{u}_i^n - u_i^{n-1} ) d\Gamma_u,
        // where \hat{u}_i^n denote the prescribed velocities.
        auto tNumCells = aResult.extent(0);
        Plato::ScalarMultiVectorT<ResultT> tResultWS("previous momentum forces", tNumCells, mNumDofsPerCell);
        for(auto& tPair : mMomentumBCs)
        {
            tPair.second->operator()(aWorkSets, tResultWS);
        }

        // multiply force vector by the corresponding nodal time steps
        auto tThetaOne = mThetaOne;
        auto tTimeStepWS = Plato::metadata<Plato::ScalarMultiVector>(aWorkSets.get("time steps"));
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
    (const Plato::SpatialModel & aSpatialModel,
     const Plato::WorkSets & aWorkSets,
     Plato::ScalarMultiVectorT<ResultT> & aResult)
    const override
    { return; }

private:
    void setParameters(Teuchos::ParameterList& aInputs)
    {
        if(aInputs.isSublist("Time Integration"))
        {
            auto tTimeIntegration = aInputs.sublist("Time Integration");
            mThetaTwo = tTimeIntegration.get<Plato::Scalar>("Time Step Multiplier Theta 1", 0.5);
            mThetaTwo = tTimeIntegration.get<Plato::Scalar>("Time Step Multiplier Theta 2", 0.0);
        }
    }

    void setNaturalBoundaryConditions(Teuchos::ParameterList& aInputs)
    {
        // the natural bcs are applied on the side sets where velocity essential
        // bcs are enforced. therefore, these side sets should be read by this function.
        std::unordered_map<std::string, std::vector<std::pair<Plato::OrdinalType, Plato::Scalar>>> tMap;
        if(aInputs.isSublist("Velocity Boundary Conditions") == false)
        {
            THROWERR("Velocity boundary conditions must be defined for fluid flow problems.")
        }
        auto tSublist = aInputs.sublist("Velocity Boundary Conditions");

        for (Teuchos::ParameterList::ConstIterator tItr = tSublist.begin(); tItr != tSublist.end(); ++tItr)
        {
            const Teuchos::ParameterEntry &tEntry = tSublist.entry(tItr);
            if (!tEntry.isList())
            {
                THROWERR(std::string("Error reading 'Velocity Boundary Conditions' block: Expects a parameter ")
                    + "list input with information pertaining to the velocity boundary conditions .")
            }
            const std::string& tParamListName = tSublist.name(tItr);
            Teuchos::ParameterList & tParamList = tSublist.sublist(tParamListName);

            if (tSublist.isParameter("Sides") == false)
            {
                THROWERR(std::string("Keyword 'Sides' is not define in Parameter List '") + tParamListName + "'.")
            }
            const auto tEntitySetName = tSublist.get<std::string>("Sides");
            mMomentumBCs[tEntitySetName] = std::make_shared<MomentumForces>(mSpatialDomain, tEntitySetName);
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

    static constexpr auto mNumSpatialDims        = PhysicsT::SimplexT::mNumSpatialDims;         /*!< number of spatial dimensions */
    static constexpr auto mNumNodesPerCell       = PhysicsT::SimplexT::mNumNodesPerCell;        /*!< number of nodes per cell */
    static constexpr auto mNumPressDofsPerCell   = PhysicsT::SimplexT::mNumMassDofsPerCell;     /*!< number of mass dofs per cell */
    static constexpr auto mNumTempDofsPerCell    = PhysicsT::SimplexT::mNumEnergyDofsPerCell;   /*!< number of energy dofs per cell */
    static constexpr auto mNumVelDofsPerCell     = PhysicsT::SimplexT::mNumMomentumDofsPerCell; /*!< number of momentum dofs per cell */
    static constexpr auto mNumControlDofsPerCell = PhysicsT::SimplexT::mNumControlDofsPerCell;  /*!< number of design variable per cell */
    static constexpr auto mNumPressDofsPerNode   = PhysicsT::SimplexT::mNumMassDofsPerNode;     /*!< number of mass dofs per node */
    static constexpr auto mNumTempDofsPerNode    = PhysicsT::SimplexT::mNumEnergyDofsPerNode;   /*!< number of energy dofs per node */
    static constexpr auto mNumVelDofsPerNode     = PhysicsT::SimplexT::mNumMomentumDofsPerNode; /*!< number of momentum dofs per node */
    static constexpr auto mNumControlDofsPerNode = PhysicsT::SimplexT::mNumControlDofsPerNode;  /*!< number of design variable per node */

    static constexpr auto mNumConfigDofsPerNode = PhysicsT::SimplexT::mNumConfigDofsPerNode; /*!< number of configuration degrees of freedom per cell */
    static constexpr auto mNumConfigDofsPerCell = PhysicsT::SimplexT::mNumConfigDofsPerCell; /*!< number of configuration degrees of freedom per cell */

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
    using ResidualFuncT      = std::shared_ptr<Plato::FluidMechanics::AbstractVectorFunction<PhysicsT, ResidualEvalT>>;
    using GradConfigFuncT    = std::shared_ptr<Plato::FluidMechanics::AbstractVectorFunction<PhysicsT, GradConfigEvalT>>;
    using GradControlFuncT   = std::shared_ptr<Plato::FluidMechanics::AbstractVectorFunction<PhysicsT, GradControlEvalT>>;
    using GradCurVelFuncT    = std::shared_ptr<Plato::FluidMechanics::AbstractVectorFunction<PhysicsT, GradCurVelEvalT>>;
    using GradPrevVelFuncT   = std::shared_ptr<Plato::FluidMechanics::AbstractVectorFunction<PhysicsT, GradPrevVelEvalT>>;
    using GradCurTempFuncT   = std::shared_ptr<Plato::FluidMechanics::AbstractVectorFunction<PhysicsT, GradCurTempEvalT>>;
    using GradPrevTempFuncT  = std::shared_ptr<Plato::FluidMechanics::AbstractVectorFunction<PhysicsT, GradPrevTempEvalT>>;
    using GradCurPressFuncT  = std::shared_ptr<Plato::FluidMechanics::AbstractVectorFunction<PhysicsT, GradCurPressEvalT>>;
    using GradPrevPressFuncT = std::shared_ptr<Plato::FluidMechanics::AbstractVectorFunction<PhysicsT, GradPrevPressEvalT>>;
    using GradPredictorFuncT = std::shared_ptr<Plato::FluidMechanics::AbstractVectorFunction<PhysicsT, GradPredictorEvalT>>;

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
    Plato::LocalOrdinalMaps<PhysicsT> mLocalOrdinalMaps;

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
        mLocalOrdinalMaps(aSpatialModel.Mesh)
    {
        this->initialize(aName, aDataMap, aInputs);
    }

    Plato::ScalarVector value
    (const Plato::ScalarVector & aControls,
     const Plato::Primal       & aVariables) const
    {
        using ResultScalarT = typename ResidualEvalT::ResultScalarType;

        auto tNumNodes = mSpatialModel.Mesh.nverts();
        auto tLength = tNumNodes * mNumDofsPerNode;
        Plato::ScalarVector tReturnValue("Assembled Residual", tLength);

        // evaluate internal forces
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            Plato::WorkSets tInputWorkSets;
            auto tNumCells = tDomain.numCells();
            Plato::FluidMechanics::build_vector_function_worksets<ResidualEvalT>
                (tDomain, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            // evaluate internal forces
            auto tName = tDomain.getDomainName();
            Plato::ScalarMultiVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells, mNumDofsPerCell);
            mResidualFuncs.at(tName)->evaluate(tInputWorkSets, tResultWS);
            Plato::assemble_residual<mNumNodesPerCell, mNumDofsPerNode>
                (tNumCells, mLocalOrdinalMaps.mVectorStateOrdinalMap, tResultWS, tReturnValue);
        }

        // evaluate boundary forces
        {
            Plato::WorkSets tInputWorkSets;
            auto tNumCells = mSpatialModel.Mesh.nelems();
            Plato::FluidMechanics::build_vector_function_worksets<ResidualEvalT>
                (tNumCells, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            // evaluate prescribed forces
            Plato::ScalarMultiVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells, mNumDofsPerCell);
            mResidualFuncs.begin()->second->evaluatePrescribed(mSpatialModel, tInputWorkSets, tResultWS);
            Plato::assemble_residual<mNumNodesPerCell, mNumDofsPerNode>
                (tNumCells, mLocalOrdinalMaps.mVectorStateOrdinalMap, tResultWS, tReturnValue);

            // evaluate balancing forces
            Plato::blas2::fill(0.0, tResultWS);
            mResidualFuncs.begin()->second->evaluateBoundary(mSpatialModel, tInputWorkSets, tResultWS);
            Plato::assemble_residual<mNumNodesPerCell, mNumDofsPerNode>
                (tNumCells, mLocalOrdinalMaps.mVectorStateOrdinalMap, tResultWS, tReturnValue);
        }

        return tReturnValue;
    }

    Teuchos::RCP<Plato::CrsMatrixType>
    gradientConfig
    (const Plato::ScalarVector & aControls,
     const Plato::Primal       & aVariables) const
    {
        using ResultScalarT = typename GradConfigEvalT::ResultScalarType;

        // create return matrix
        auto tMesh = mSpatialModel.Mesh;
        Teuchos::RCP<Plato::CrsMatrixType> tJacobian =
            Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumConfigDofsPerNode, mNumDofsPerNode>(&tMesh);

        // evaluate internal forces
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            Plato::WorkSets tInputWorkSets;
            auto tNumCells = tDomain.numCells();
            Plato::FluidMechanics::build_vector_function_worksets<GradConfigEvalT>
                (tDomain, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            // evaluate internal forces
            auto tName = tDomain.getDomainName();
            Plato::ScalarMultiVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells, mNumDofsPerCell);
            mGradConfigFuncs.at(tName)->evaluate(tInputWorkSets, tResultWS);
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumConfigDofsPerNode, mNumDofsPerNode> tJacEntryOrdinal(tJacobian, &tMesh);
            Plato::assemble_transpose_jacobian(tDomain, mNumDofsPerCell, mNumConfigDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobian->entries());
        }

        // evaluate boundary forces
        {
            Plato::WorkSets tInputWorkSets;
            auto tNumCells = mSpatialModel.Mesh.nelems();
            Plato::FluidMechanics::build_vector_function_worksets<GradConfigEvalT>
                (tNumCells, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            // evaluate prescribed forces
            Plato::ScalarMultiVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells, mNumDofsPerCell);
            mGradConfigFuncs.begin()->second->evaluatePrescribed(mSpatialModel, tInputWorkSets, tResultWS);
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumConfigDofsPerNode, mNumDofsPerNode> tJacEntryOrdinal(tJacobian, &tMesh);
            Plato::assemble_transpose_jacobian(tNumCells, mNumDofsPerCell, mNumConfigDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobian->entries());

            // evaluate balancing forces
            Plato::blas2::fill(0.0, tResultWS);
            mGradConfigFuncs.begin()->second->evaluateBoundary(mSpatialModel, tInputWorkSets, tResultWS);
            Plato::assemble_transpose_jacobian(tNumCells, mNumDofsPerCell, mNumConfigDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobian->entries());
        }

        return tJacobian;
    }

    Teuchos::RCP<Plato::CrsMatrixType>
    gradientControl
    (const Plato::ScalarVector & aControls,
     const Plato::Primal       & aVariables) const
    {
        using ResultScalarT = typename GradControlEvalT::ResultScalarType;

        // create return matrix
        auto tMesh = mSpatialModel.Mesh;
        Teuchos::RCP<Plato::CrsMatrixType> tJacobian =
            Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumControlDofsPerNode, mNumDofsPerNode>( &tMesh );

        // evaluate internal forces
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            Plato::WorkSets tInputWorkSets;
            auto tNumCells = tDomain.numCells();
            Plato::FluidMechanics::build_vector_function_worksets<GradControlEvalT>
                (tDomain, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            // evaluate internal forces
            auto tName = tDomain.getDomainName();
            Plato::ScalarMultiVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells, mNumDofsPerCell);
            mGradControlFuncs.at(tName)->evaluate(tInputWorkSets, tResultWS);
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumControlDofsPerNode, mNumDofsPerNode> tJacEntryOrdinal(tJacobian, &tMesh);
            Plato::assemble_transpose_jacobian(tDomain, mNumDofsPerCell, mNumControlDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobian->entries());
        }

        // evaluate boundary forces
        {
            Plato::WorkSets tInputWorkSets;
            auto tNumCells = mSpatialModel.Mesh.nelems();
            Plato::FluidMechanics::build_vector_function_worksets<GradControlEvalT>
                (tNumCells, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            // evaluate prescribed forces
            Plato::ScalarMultiVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells, mNumDofsPerCell);
            mGradControlFuncs.begin()->second->evaluatePrescribed(mSpatialModel, tInputWorkSets, tResultWS);
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumControlDofsPerNode, mNumDofsPerNode> tJacEntryOrdinal(tJacobian, &tMesh);
            Plato::assemble_transpose_jacobian(tNumCells, mNumDofsPerCell, mNumControlDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobian->entries());

            // evaluate balancing forces
            Plato::blas2::fill(0.0, tResultWS);
            mGradControlFuncs.begin()->second->evaluateBoundary(mSpatialModel, tInputWorkSets, tResultWS);
            Plato::assemble_transpose_jacobian(tNumCells, mNumDofsPerCell, mNumControlDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobian->entries());
        }

        return tJacobian;
    }

    Teuchos::RCP<Plato::CrsMatrixType>
    gradientPredictor
    (const Plato::ScalarVector & aControls,
     const Plato::Primal       & aVariables) const
    {
        using ResultScalarT = typename GradPredictorEvalT::ResultScalarType;

        // create return matrix
        auto tMesh = mSpatialModel.Mesh;
        Teuchos::RCP<Plato::CrsMatrixType> tJacobian =
            Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumVelDofsPerNode, mNumDofsPerNode>( &tMesh );

        // evaluate internal forces
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            Plato::WorkSets tInputWorkSets;
            auto tNumCells = tDomain.numCells();
            Plato::FluidMechanics::build_vector_function_worksets<GradPredictorEvalT>
                (tDomain, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            // evaluate internal forces
            auto tName = tDomain.getDomainName();
            Plato::ScalarMultiVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells, mNumDofsPerCell);
            mGradPredictorFuncs.at(tName)->evaluate(tInputWorkSets, tResultWS);
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumVelDofsPerNode, mNumDofsPerNode> tJacEntryOrdinal(tJacobian, &tMesh);
            Plato::assemble_transpose_jacobian(tDomain, mNumDofsPerCell, mNumVelDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobian->entries());
        }

        // evaluate boundary forces
        {
            Plato::WorkSets tInputWorkSets;
            auto tNumCells = mSpatialModel.Mesh.nelems();
            Plato::FluidMechanics::build_vector_function_worksets<GradPredictorEvalT>
                (tNumCells, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            // evaluate prescribed forces
            Plato::ScalarMultiVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells, mNumDofsPerCell);
            mGradPredictorFuncs.begin()->second->evaluatePrescribed(mSpatialModel, tInputWorkSets, tResultWS);
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumVelDofsPerNode, mNumDofsPerNode> tJacEntryOrdinal(tJacobian, &tMesh);
            Plato::assemble_transpose_jacobian(tNumCells, mNumDofsPerCell, mNumVelDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobian->entries());

            // evaluate balancing forces
            Plato::blas2::fill(0.0, tResultWS);
            mGradPredictorFuncs.begin()->second->evaluateBoundary(mSpatialModel, tInputWorkSets, tResultWS);
            Plato::assemble_transpose_jacobian(tNumCells, mNumDofsPerCell, mNumVelDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobian->entries());
        }

        return tJacobian;
    }

    Teuchos::RCP<Plato::CrsMatrixType>
    gradientPreviousVel
    (const Plato::ScalarVector & aControls,
     const Plato::Primal       & aVariables) const
    {
        using ResultScalarT = typename GradPrevVelEvalT::ResultScalarType;

        // create return matrix
        auto tMesh = mSpatialModel.Mesh;
        Teuchos::RCP<Plato::CrsMatrixType> tJacobian =
            Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumVelDofsPerNode, mNumDofsPerNode>( &tMesh );

        // evaluate internal forces
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            Plato::WorkSets tInputWorkSets;
            auto tNumCells = tDomain.numCells();
            Plato::FluidMechanics::build_vector_function_worksets<GradPrevVelEvalT>
                (tDomain, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            // evaluate internal forces
            auto tName = tDomain.getDomainName();
            Plato::ScalarMultiVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells, mNumDofsPerCell);
            mGradPrevVelFuncs.at(tName)->evaluate(tInputWorkSets, tResultWS);
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumVelDofsPerNode, mNumDofsPerNode> tJacEntryOrdinal(tJacobian, &tMesh);
            Plato::assemble_transpose_jacobian(tDomain, mNumDofsPerCell, mNumVelDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobian->entries());
        }

        // evaluate boundary forces
        {
            Plato::WorkSets tInputWorkSets;
            auto tNumCells = mSpatialModel.Mesh.nelems();
            Plato::FluidMechanics::build_vector_function_worksets<GradPrevVelEvalT>
                (tNumCells, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            // evaluate prescribed forces
            Plato::ScalarMultiVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells, mNumDofsPerCell);
            mGradPrevVelFuncs.begin()->second->evaluatePrescribed(mSpatialModel, tInputWorkSets, tResultWS);
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumVelDofsPerNode, mNumDofsPerNode> tJacEntryOrdinal(tJacobian, &tMesh);
            Plato::assemble_transpose_jacobian(tNumCells, mNumDofsPerCell, mNumVelDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobian->entries());

            // evaluate balancing forces
            Plato::blas2::fill(0.0, tResultWS);
            mGradPrevVelFuncs.begin()->second->evaluateBoundary(mSpatialModel, tInputWorkSets, tResultWS);
            Plato::assemble_transpose_jacobian(tNumCells, mNumDofsPerCell, mNumVelDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobian->entries());
        }

        return tJacobian;
    }

    Teuchos::RCP<Plato::CrsMatrixType>
    gradientPreviousPress
    (const Plato::ScalarVector & aControls,
     const Plato::Primal       & aVariables) const
    {
        using ResultScalarT = typename GradPrevPressEvalT::ResultScalarType;

        // create return matrix
        auto tMesh = mSpatialModel.Mesh;
        Teuchos::RCP<Plato::CrsMatrixType> tJacobian =
            Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumPressDofsPerNode, mNumDofsPerNode>( &tMesh );

        // evaluate internal forces
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            Plato::WorkSets tInputWorkSets;
            auto tNumCells = tDomain.numCells();
            Plato::FluidMechanics::build_vector_function_worksets<GradPrevPressEvalT>
                (tDomain, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            // evaluate internal forces
            auto tName = tDomain.getDomainName();
            Plato::ScalarMultiVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells, mNumDofsPerCell);
            mGradPrevPressFuncs.at(tName)->evaluate(tInputWorkSets, tResultWS);
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumPressDofsPerNode, mNumDofsPerNode> tJacEntryOrdinal(tJacobian, &tMesh);
            Plato::assemble_transpose_jacobian(tDomain, mNumDofsPerCell, mNumPressDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobian->entries());
        }

        // evaluate boundary forces
        {
            Plato::WorkSets tInputWorkSets;
            auto tNumCells = mSpatialModel.Mesh.nelems();
            Plato::FluidMechanics::build_vector_function_worksets<GradPrevPressEvalT>
                (tNumCells, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            // evaluate prescribed forces
            Plato::ScalarMultiVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells, mNumDofsPerCell);
            mGradPrevPressFuncs.begin()->second->evaluatePrescribed(mSpatialModel, tInputWorkSets, tResultWS);
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumPressDofsPerNode, mNumDofsPerNode> tJacEntryOrdinal(tJacobian, &tMesh);
            Plato::assemble_transpose_jacobian(tNumCells, mNumDofsPerCell, mNumPressDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobian->entries());

            // evaluate balancing forces
            Plato::blas2::fill(0.0, tResultWS);
            mGradPrevPressFuncs.begin()->second->evaluateBoundary(mSpatialModel, tInputWorkSets, tResultWS);
            Plato::assemble_transpose_jacobian(tNumCells, mNumDofsPerCell, mNumPressDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobian->entries());
        }

        return tJacobian;
    }

    Teuchos::RCP<Plato::CrsMatrixType>
    gradientPreviousTemp
    (const Plato::ScalarVector & aControls,
     const Plato::Primal       & aVariables) const
    {
        using ResultScalarT = typename GradPrevTempEvalT::ResultScalarType;

        // create return matrix
        auto tMesh = mSpatialModel.Mesh;
        Teuchos::RCP<Plato::CrsMatrixType> tJacobian =
            Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumTempDofsPerNode, mNumDofsPerNode>( &tMesh );

        // evaluate internal forces
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            Plato::WorkSets tInputWorkSets;
            auto tNumCells = tDomain.numCells();
            Plato::FluidMechanics::build_vector_function_worksets<GradPrevTempEvalT>
                (tDomain, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            // evaluate internal forces
            auto tName = tDomain.getDomainName();
            Plato::ScalarMultiVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells, mNumDofsPerCell);
            mGradPrevTempFuncs.at(tName)->evaluate(tInputWorkSets, tResultWS);
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumTempDofsPerNode, mNumDofsPerNode> tJacEntryOrdinal(tJacobian, &tMesh);
            Plato::assemble_transpose_jacobian(tDomain, mNumDofsPerCell, mNumTempDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobian->entries());
        }

        // evaluate boundary forces
        {
            Plato::WorkSets tInputWorkSets;
            auto tNumCells = mSpatialModel.Mesh.nelems();
            Plato::FluidMechanics::build_vector_function_worksets<GradPrevTempEvalT>
                (tNumCells, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            // evaluate prescribed forces
            Plato::ScalarMultiVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells, mNumDofsPerCell);
            mGradPrevTempFuncs.begin()->second->evaluatePrescribed(mSpatialModel, tInputWorkSets, tResultWS);
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumTempDofsPerNode, mNumDofsPerNode> tJacEntryOrdinal(tJacobian, &tMesh);
            Plato::assemble_transpose_jacobian(tNumCells, mNumDofsPerCell, mNumTempDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobian->entries());

            // evaluate balancing forces
            Plato::blas2::fill(0.0, tResultWS);
            mGradPrevTempFuncs.begin()->second->evaluateBoundary(mSpatialModel, tInputWorkSets, tResultWS);
            Plato::assemble_transpose_jacobian(tNumCells, mNumDofsPerCell, mNumTempDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobian->entries());
        }

        return tJacobian;
    }

    Teuchos::RCP<Plato::CrsMatrixType>
    gradientCurrentVel
    (const Plato::ScalarVector & aControls,
     const Plato::Primal       & aVariables) const
    {
        using ResultScalarT = typename GradCurVelEvalT::ResultScalarType;

        // create return matrix
        auto tMesh = mSpatialModel.Mesh;
        Teuchos::RCP<Plato::CrsMatrixType> tJacobian =
            Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumVelDofsPerNode, mNumDofsPerNode>( &tMesh );

        // evaluate internal forces
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            Plato::WorkSets tInputWorkSets;
            auto tNumCells = tDomain.numCells();
            Plato::FluidMechanics::build_vector_function_worksets<GradCurVelEvalT>
                (tDomain, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            // evaluate internal forces
            auto tName = tDomain.getDomainName();
            Plato::ScalarMultiVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells, mNumDofsPerCell);
            mGradCurVelFuncs.at(tName)->evaluate(tInputWorkSets, tResultWS);
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumVelDofsPerNode, mNumDofsPerNode> tJacEntryOrdinal(tJacobian, &tMesh);
            Plato::assemble_transpose_jacobian(tDomain, mNumDofsPerCell, mNumVelDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobian->entries());
        }

        // evaluate boundary forces
        {
            Plato::WorkSets tInputWorkSets;
            auto tNumCells = mSpatialModel.Mesh.nelems();
            Plato::FluidMechanics::build_vector_function_worksets<GradCurVelEvalT>
                (tNumCells, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            // evaluate prescribed forces
            Plato::ScalarMultiVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells, mNumDofsPerCell);
            mGradCurVelFuncs.begin()->second->evaluatePrescribed(mSpatialModel, tInputWorkSets, tResultWS);
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumVelDofsPerNode, mNumDofsPerNode> tJacEntryOrdinal(tJacobian, &tMesh);
            Plato::assemble_transpose_jacobian(tNumCells, mNumDofsPerCell, mNumVelDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobian->entries());

            // evaluate balancing forces
            Plato::blas2::fill(0.0, tResultWS);
            mGradCurVelFuncs.begin()->second->evaluateBoundary(mSpatialModel, tInputWorkSets, tResultWS);
            Plato::assemble_transpose_jacobian(tNumCells, mNumDofsPerCell, mNumVelDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobian->entries());
        }

        return tJacobian;
    }

    Teuchos::RCP<Plato::CrsMatrixType>
    gradientCurrentPress
    (const Plato::ScalarVector & aControls,
     const Plato::Primal       & aVariables) const
    {
        using ResultScalarT = typename GradCurPressEvalT::ResultScalarType;

        // create return matrix
        auto tMesh = mSpatialModel.Mesh;
        Teuchos::RCP<Plato::CrsMatrixType> tJacobian =
            Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumPressDofsPerNode, mNumDofsPerNode>( &tMesh );

        // evaluate internal forces
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            Plato::WorkSets tInputWorkSets;
            auto tNumCells = tDomain.numCells();
            Plato::FluidMechanics::build_vector_function_worksets<GradCurPressEvalT>
                (tDomain, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            // evaluate internal forces
            auto tName = tDomain.getDomainName();
            Plato::ScalarMultiVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells, mNumDofsPerCell);
            mGradCurPressFuncs.at(tName)->evaluate(tInputWorkSets, tResultWS);
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumPressDofsPerNode, mNumDofsPerNode> tJacEntryOrdinal(tJacobian, &tMesh);
            Plato::assemble_transpose_jacobian(tDomain, mNumDofsPerCell, mNumPressDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobian->entries());
        }

        // evaluate boundary forces
        {
            Plato::WorkSets tInputWorkSets;
            auto tNumCells = mSpatialModel.Mesh.nelems();
            Plato::FluidMechanics::build_vector_function_worksets<GradCurPressEvalT>
                (tNumCells, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            // evaluate prescribed forces
            Plato::ScalarMultiVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells, mNumDofsPerCell);
            mGradCurPressFuncs.begin()->second->evaluatePrescribed(mSpatialModel, tInputWorkSets, tResultWS);
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumPressDofsPerNode, mNumDofsPerNode> tJacEntryOrdinal(tJacobian, &tMesh);
            Plato::assemble_transpose_jacobian(tNumCells, mNumDofsPerCell, mNumPressDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobian->entries());

            // evaluate balancing forces
            Plato::blas2::fill(0.0, tResultWS);
            mGradCurPressFuncs.begin()->second->evaluateBoundary(mSpatialModel, tInputWorkSets, tResultWS);
            Plato::assemble_transpose_jacobian(tNumCells, mNumDofsPerCell, mNumPressDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobian->entries());
        }

        return tJacobian;
    }

    Teuchos::RCP<Plato::CrsMatrixType>
    gradientCurrentTemp
    (const Plato::ScalarVector & aControls,
     const Plato::Primal       & aVariables) const
    {
        using ResultScalarT = typename GradCurTempEvalT::ResultScalarType;

        // create return matrix
        auto tMesh = mSpatialModel.Mesh;
        Teuchos::RCP<Plato::CrsMatrixType> tJacobian =
            Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumTempDofsPerNode, mNumDofsPerNode>( &tMesh );

        // evaluate internal forces
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            Plato::WorkSets tInputWorkSets;
            auto tNumCells = tDomain.numCells();
            Plato::FluidMechanics::build_vector_function_worksets<GradCurTempEvalT>
                (tDomain, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            // evaluate internal forces
            auto tName = tDomain.getDomainName();
            Plato::ScalarMultiVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells, mNumDofsPerCell);
            mGradCurTempFuncs.at(tName)->evaluate(tInputWorkSets, tResultWS);
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumTempDofsPerNode, mNumDofsPerNode> tJacEntryOrdinal(tJacobian, &tMesh);
            Plato::assemble_transpose_jacobian(tDomain, mNumDofsPerCell, mNumTempDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobian->entries());
        }

        // evaluate prescribed forces
        {
            Plato::WorkSets tInputWorkSets;
            auto tNumCells = mSpatialModel.Mesh.nelems();
            Plato::FluidMechanics::build_vector_function_worksets<GradCurTempEvalT>
                (tNumCells, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            // evaluate boundary forces
            Plato::ScalarMultiVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells, mNumDofsPerCell);
            mGradCurTempFuncs.begin()->second->evaluatePrescribed(mSpatialModel, tInputWorkSets, tResultWS);
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumTempDofsPerNode, mNumDofsPerNode> tJacEntryOrdinal(tJacobian, &tMesh);
            Plato::assemble_transpose_jacobian(tNumCells, mNumDofsPerCell, mNumTempDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobian->entries());

            // evaluate balancing forces
            Plato::blas2::fill(0.0, tResultWS);
            mGradCurTempFuncs.begin()->second->evaluateBoundary(mSpatialModel, tInputWorkSets, tResultWS);
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

            mResidualFuncs[tName]  = tVecFuncFactory.template createVectorFunction<PhysicsT, ResidualEvalT>
                (aName, tDomain, aDataMap, aInputs);

            mGradControlFuncs[tName] = tVecFuncFactory.template createVectorFunction<PhysicsT, GradControlEvalT>
                (aName, tDomain, aDataMap, aInputs);

            mGradConfigFuncs[tName] = tVecFuncFactory.template createVectorFunction<PhysicsT, GradConfigEvalT>
                (aName, tDomain, aDataMap, aInputs);

            mGradCurPressFuncs[tName] = tVecFuncFactory.template createVectorFunction<PhysicsT, GradCurPressEvalT>
                (aName, tDomain, aDataMap, aInputs);

            mGradPrevPressFuncs[tName] = tVecFuncFactory.template createVectorFunction<PhysicsT, GradPrevPressEvalT>
                (aName, tDomain, aDataMap, aInputs);

            mGradCurTempFuncs[tName] = tVecFuncFactory.template createVectorFunction<PhysicsT, GradCurTempEvalT>
                (aName, tDomain, aDataMap, aInputs);

            mGradPrevTempFuncs[tName] = tVecFuncFactory.template createVectorFunction<PhysicsT, GradPrevTempEvalT>
                (aName, tDomain, aDataMap, aInputs);

            mGradCurVelFuncs[tName] = tVecFuncFactory.template createVectorFunction<PhysicsT, GradCurVelEvalT>
                (aName, tDomain, aDataMap, aInputs);

            mGradPrevVelFuncs[tName] = tVecFuncFactory.template createVectorFunction<PhysicsT, GradPrevVelEvalT>
                (aName, tDomain, aDataMap, aInputs);

            mGradPredictorFuncs[tName] = tVecFuncFactory.template createVectorFunction<PhysicsT, GradPredictorEvalT>
                (aName, tDomain, aDataMap, aInputs);
        }
    }
};
// class VectorFunction

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
        auto tLowerTag = Plato::tolower(aTag);
        // TODO: explore function interface for constructor, similar to how it is done in the xml generator
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
    (const std::string          & aName,
     const Plato::SpatialDomain & aDomain,
     Plato::DataMap             & aDataMap,
     Teuchos::ParameterList     & aInputs)
    {
        if( !aInputs.isSublist("Criteria") )
        {
            THROWERR("'Criteria' block is not defined.")
        }
        auto tCriteriaList = aInputs.sublist("Criteria");
        if( !tCriteriaList.isSublist(aName) )
        {
            THROWERR(std::string("Criteria Block with name '") + aName + "' is not defined.")
        }
        auto tCriterion = tCriteriaList.sublist(aName);

        if(!tCriterion.isParameter("Scalar Function Type"))
        {
            THROWERR(std::string("'Scalar Function Type' keyword is not defined in Criterion with name '") + aName + "'.")
        }

        auto tFlowTag = tCriterion.get<std::string>("Flow", "Not Defined");
        auto tFlowLowerTag = Plato::tolower(tFlowTag);
        auto tCriterionTag = tCriterion.get<std::string>("Scalar Function Type", "Not Defined");
        auto tCriterionLowerTag = Plato::tolower(tCriterionTag);

        if( tCriterionLowerTag == "average surface pressure" )
        {
            return ( std::make_shared<Plato::FluidMechanics::AverageSurfacePressure<PhysicsT, EvaluationT>>
                (aName, aDomain, aDataMap, aInputs) );
        }
        else if( tCriterionLowerTag == "average surface temperature" )
        {
            return ( std::make_shared<Plato::FluidMechanics::AverageSurfaceTemperature<PhysicsT, EvaluationT>>
                (aName, aDomain, aDataMap, aInputs) );
        }
        else if( tCriterionLowerTag == "internal dissipation energy" && tFlowLowerTag == "incompressible")
        {
            return ( std::make_shared<Plato::FluidMechanics::InternalDissipationEnergyIncompressible<PhysicsT, EvaluationT>>
                (aName, aDomain, aDataMap, aInputs) );
        }
        else
        {
            THROWERR(std::string("'Scalar Function Type' with tag '") + tCriterionTag
                + "' in Criterion Block '" + aName + "' is not supported.")
        }
    }
};
// struct FunctionFactory


// todo: criterion factory
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
        /*else if(tLowerType == "weighted sum")
        {
            auto tCriterion =
                std::make_shared<Plato::FluidMechanics::WeightedScalarFunction<PhysicsT>>
                    (aSpatialModel, aDataMap, aInputs, aName);
            return tCriterion;
        }
        else if(tLowerType == "least squares")
        {
            auto tCriterion =
                std::make_shared<Plato::FluidMechanics::LeastSquares<PhysicsT>>
                    (aSpatialModel, aDataMap, aInputs, aName);
            return tCriterion;
        }*/
        else
        {
            THROWERR(std::string("Scalar function in block '") + aName + "' with type '" + tType + "' is not supported.")
        }
     }
};


// todo: finish weighted scalar function
template<typename PhysicsT>
class WeightedScalarFunction : public Plato::FluidMechanics::CriterionBase
{
private:
    // static metadata
    static constexpr auto mNumSpatialDims        = PhysicsT::SimplexT::mNumSpatialDims;         /*!< number of spatial dimensions */
    static constexpr auto mNumPressDofsPerNode   = PhysicsT::SimplexT::mNumMassDofsPerNode;     /*!< number of mass dofs per node */
    static constexpr auto mNumTempDofsPerNode    = PhysicsT::SimplexT::mNumEnergyDofsPerNode;   /*!< number of energy dofs per node */
    static constexpr auto mNumVelDofsPerNode     = PhysicsT::SimplexT::mNumMomentumDofsPerNode; /*!< number of momentum dofs per node */
    static constexpr auto mNumControlDofsPerNode = PhysicsT::SimplexT::mNumControlDofsPerNode;  /*!< number of design variables per node */

    // set local typenames
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
     const Plato::Primal & aVariables) const override
    {
        Plato::Scalar tResult = 0.0;
        for (auto& tCriterion : mCriteria)
        {
            auto tIndex = &tCriterion - &mCriteria[0];
            const auto tValue = tCriterion->value(aControls, aVariables);
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
     const Plato::Primal & aVariables) const override
    {
        const auto tNumNodes = mSpatialModel.Mesh.nverts();
        Plato::ScalarVector tTotalDerivative("total derivative", mNumSpatialDims * tNumNodes);
        for (auto& tCriterion : mCriteria)
        {
            auto tIndex = &tCriterion - &mCriteria[0];
            auto tGradient = tCriterion->gradientConfig(aControls, aVariables);
            const auto tFuncWeight = mCriterionWeights[tIndex];
            Plato::blas1::update(tFuncWeight, tGradient, static_cast<Plato::Scalar>(1.0), tTotalDerivative);
        }
        return tTotalDerivative;
    }

    Plato::ScalarVector gradientControl
    (const Plato::ScalarVector & aControls,
     const Plato::Primal & aVariables) const override
    {
        const auto tNumNodes = mSpatialModel.Mesh.nverts();
        Plato::ScalarVector tTotalDerivative("total derivative", mNumControlDofsPerNode * tNumNodes);
        for (auto& tCriterion : mCriteria)
        {
            auto tIndex = &tCriterion - &mCriteria[0];
            auto tGradient = tCriterion->gradientControl(aControls, aVariables);
            const auto tFuncWeight = mCriterionWeights[tIndex];
            Plato::blas1::update(tFuncWeight, tGradient, static_cast<Plato::Scalar>(1.0), tTotalDerivative);
        }
        return tTotalDerivative;
    }

    Plato::ScalarVector gradientCurrentPress
    (const Plato::ScalarVector & aControls,
     const Plato::Primal & aVariables) const override
    {
        const auto tNumNodes = mSpatialModel.Mesh.nverts();
        Plato::ScalarVector tTotalDerivative("total derivative", mNumPressDofsPerNode * tNumNodes);
        for (auto& tCriterion : mCriteria)
        {
            auto tIndex = &tCriterion - &mCriteria[0];
            auto tGradient = tCriterion->gradientCurrentPress(aControls, aVariables);
            const auto tFuncWeight = mCriterionWeights[tIndex];
            Plato::blas1::update(tFuncWeight, tGradient, static_cast<Plato::Scalar>(1.0), tTotalDerivative);
        }
        return tTotalDerivative;
    }

    Plato::ScalarVector gradientCurrentTemp
    (const Plato::ScalarVector & aControls,
     const Plato::Primal & aVariables) const override
    {
        const auto tNumNodes = mSpatialModel.Mesh.nverts();
        Plato::ScalarVector tTotalDerivative("total derivative", mNumTempDofsPerNode * tNumNodes);
        for (auto& tCriterion : mCriteria)
        {
            auto tIndex = &tCriterion - &mCriteria[0];
            auto tGradient = tCriterion->gradientCurrentTemp(aControls, aVariables);
            const auto tFuncWeight = mCriterionWeights[tIndex];
            Plato::blas1::update(tFuncWeight, tGradient, static_cast<Plato::Scalar>(1.0), tTotalDerivative);
        }
        return tTotalDerivative;
    }

    Plato::ScalarVector gradientCurrentVel
    (const Plato::ScalarVector & aControls,
     const Plato::Primal & aVariables) const override
    {
        const auto tNumNodes = mSpatialModel.Mesh.nverts();
        Plato::ScalarVector tTotalDerivative("total derivative", mNumVelDofsPerNode * tNumNodes);
        for (auto& tCriterion : mCriteria)
        {
            auto tIndex = &tCriterion - &mCriteria[0];
            auto tGradient = tCriterion->gradientCurrentVel(aControls, aVariables);
            const auto tFuncWeight = mCriterionWeights[tIndex];
            Plato::blas1::update(tFuncWeight, tGradient, static_cast<Plato::Scalar>(1.0), tTotalDerivative);
        }
        return tTotalDerivative;
    }

private:
    void checkInputs()
    {
        if (mCriterionNames.size() != mCriterionWeights.size())
        {
            THROWERR(std::string("Dimensions mismatch.  Number of 'Functions' and 'Weights' do not match. ") +
                     "Check scalar function with name '" + mFuncName + "'.")
        }
    }

    void initialize(Teuchos::ParameterList & aInputs)
    {
        if(aInputs.sublist("Criteria").isSublist(mFuncName) == false)
        {
            THROWERR(std::string("Scalar function with tag '") + mFuncName + "' is not defined in the input file.")
        }

        auto tCriteriaInputs = aInputs.sublist("Criteria").sublist(mFuncName);
        this->parseFunctions(tCriteriaInputs);
        this->parseWeights(tCriteriaInputs);
        this->checkInputs();

        Plato::FluidMechanics::CriterionFactory<PhysicsT> tFactory;
        for(auto& tName : mCriterionNames)
        {
            auto tScalarFunction = tFactory.createCriterion(mSpatialModel, mDataMap, aInputs, tName);
            mCriteria.push_back(tScalarFunction);
        }
    }

    void parseFunctions(Teuchos::ParameterList & aInputs)
    {
        mCriterionNames = Plato::parse_array<std::string>("Functions", aInputs);
        if(mCriterionNames.empty())
        {
            THROWERR(std::string("'Functions' keyword was not defined in function block with name '") + mFuncName
                + "'. User must define 'functions' used in a 'Weighted Sum' criterion.")
        }
    }

    void parseWeights(Teuchos::ParameterList & aInputs)
    {
        mCriterionWeights = Plato::parse_array<Plato::Scalar>("Weights", aInputs);
        if(mCriterionNames.empty())
        {
            THROWERR(std::string("'Weights' keyword was not defined in function block with name '") + mFuncName
                + "'. User must define 'weights' used in a 'Weighted Sum' criterion.")
        }
    }
};
// class WeightedScalarFunction





// todo: least squares scalar function
template<typename PhysicsT>
class LeastSquaresScalarFunction : public Plato::FluidMechanics::CriterionBase
{
private:
    // static metadata
    static constexpr auto mNumSpatialDims        = PhysicsT::SimplexT::mNumSpatialDims;         /*!< number of spatial dimensions */
    static constexpr auto mNumPressDofsPerNode   = PhysicsT::SimplexT::mNumMassDofsPerNode;     /*!< number of mass dofs per node */
    static constexpr auto mNumTempDofsPerNode    = PhysicsT::SimplexT::mNumEnergyDofsPerNode;   /*!< number of energy dofs per node */
    static constexpr auto mNumVelDofsPerNode     = PhysicsT::SimplexT::mNumMomentumDofsPerNode; /*!< number of momentum dofs per node */
    static constexpr auto mNumControlDofsPerNode = PhysicsT::SimplexT::mNumControlDofsPerNode;  /*!< number of control dofs per node */
    static constexpr auto mNumConfigDofsPerNode  = PhysicsT::SimplexT::mNumConfigDofsPerNode;   /*!< number of configuration dofs per node */

    // set local typenames
    using Criterion = std::shared_ptr<Plato::FluidMechanics::CriterionBase>;

    bool mDiagnostics; /*!< write diagnostics to terminal */
    Plato::DataMap& mDataMap; /*!< output database */
    const Plato::SpatialModel& mSpatialModel; /*!< mesh database */

    std::string mFuncName; /*!< weighted scalar function name */

    std::vector<Criterion>     mCriteria;                /*!< list of scalar function criteria */
    std::vector<std::string>   mCriterionNames;          /*!< list of criterion names */
    std::vector<Plato::Scalar> mCriterionWeights;        /*!< list of criterion weights */
    std::vector<Plato::Scalar> mCriterionNormalizations; /*!< list of criterion normalization */

public:
    LeastSquaresScalarFunction
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

    std::string name() const override
    {
        return mFuncName;
    }

    Plato::Scalar value
    (const Plato::ScalarVector & aControls,
     const Plato::Primal & aVariables) const override
    {
        Plato::Scalar tResult = 0.0;
        for(auto& tCriterion : mCriteria)
        {
            auto tIndex = &tCriterion - &mCriteria[0];
            const auto tWeight = mCriterionWeights[tIndex];
            const auto tNormalization = mCriterionNormalizations[tIndex];
            auto tCriterionValue = tCriterion->value(aControls, aVariables);

            auto tValue = tCriterionValue / tNormalization;
            tResult += tWeight * (tValue * tValue);
        }
        return tResult;
    }

    Plato::ScalarVector gradientConfig
    (const Plato::ScalarVector & aControls,
     const Plato::Primal & aVariables) const override
    {
        const auto tNumDofs = mNumConfigDofsPerNode * mSpatialModel.Mesh.nverts();
        Plato::ScalarVector tGradConfig("gradient configuration", tNumDofs);
        Plato::blas1::fill(0.0, tGradConfig);
        for(auto& tCriterion : mCriteria)
        {
            auto tIndex = &tCriterion - &mCriteria[0];
            const auto tWeight = mCriterionWeights[tIndex];
            const auto tNormalization = mCriterionNormalizations[tIndex];
            auto tCriterionValue = tCriterion->value(aControls, aVariables);
            auto tCriterionGrad  = tCriterion->gradientConfig(aControls, aVariables);

            auto tMultiplier = ( static_cast<Plato::Scalar>(2.0) * tWeight * tCriterionValue )
                               / ( tNormalization * tNormalization );
            Plato::blas1::update(tMultiplier, tCriterionGrad, 1.0, tGradConfig);
        }
        return tGradConfig;
    }

    Plato::ScalarVector gradientControl
    (const Plato::ScalarVector & aControls,
     const Plato::Primal & aVariables) const override
    {
        const auto tNumDofs = mNumControlDofsPerNode * mSpatialModel.Mesh.nverts();
        Plato::ScalarVector tGradControl("gradient control", tNumDofs);
        Plato::blas1::fill(0.0, tGradControl);
        for(auto& tCriterion : mCriteria)
        {
            auto tIndex = &tCriterion - &mCriteria[0];
            const auto tWeight = mCriterionWeights[tIndex];
            const auto tNormalization = mCriterionNormalizations[tIndex];
            auto tCriterionValue = tCriterion->value(aControls, aVariables);
            auto tCriterionGrad  = tCriterion->gradientControl(aControls, aVariables);

            auto tMultiplier = ( static_cast<Plato::Scalar>(2.0) * tWeight * tCriterionValue )
                               / ( tNormalization * tNormalization );
            Plato::blas1::update(tMultiplier, tCriterionGrad, 1.0, tGradControl);
        }
        return tGradControl;
    }

    Plato::ScalarVector gradientCurrentPress
    (const Plato::ScalarVector & aControls,
     const Plato::Primal & aVariables) const override
    {
        const auto tNumDofs = mNumPressDofsPerNode * mSpatialModel.Mesh.nverts();
        Plato::ScalarVector tGradCurPress("gradient current pressure", tNumDofs);
        Plato::blas1::fill(0.0, tGradCurPress);
        for(auto& tCriterion : mCriteria)
        {
            auto tIndex = &tCriterion - &mCriteria[0];
            const auto tWeight = mCriterionWeights[tIndex];
            const auto tNormalization = mCriterionNormalizations[tIndex];
            auto tCriterionValue = tCriterion->value(aControls, aVariables);
            auto tCriterionGrad  = tCriterion->gradientCurrentPress(aControls, aVariables);

            auto tMultiplier = ( static_cast<Plato::Scalar>(2.0) * tWeight * tCriterionValue )
                               / ( tNormalization * tNormalization );
            Plato::blas1::update(tMultiplier, tCriterionGrad, 1.0, tGradCurPress);
        }
        return tGradCurPress;
    }

    Plato::ScalarVector gradientCurrentTemp
    (const Plato::ScalarVector & aControls,
     const Plato::Primal & aVariables) const override
    {
        const auto tNumDofs = mNumTempDofsPerNode * mSpatialModel.Mesh.nverts();
        Plato::ScalarVector tGradCurTemp("gradient current temperature", tNumDofs);
        Plato::blas1::fill(0.0, tGradCurTemp);
        for(auto& tCriterion : mCriteria)
        {
            auto tIndex = &tCriterion - &mCriteria[0];
            const auto tWeight = mCriterionWeights[tIndex];
            const auto tNormalization = mCriterionNormalizations[tIndex];
            auto tCriterionValue = tCriterion->value(aControls, aVariables);
            auto tCriterionGrad  = tCriterion->gradientCurrentTemp(aControls, aVariables);

            auto tMultiplier = ( static_cast<Plato::Scalar>(2.0) * tWeight * tCriterionValue )
                               / ( tNormalization * tNormalization );
            Plato::blas1::update(tMultiplier, tCriterionGrad, 1.0, tGradCurTemp);
        }
        return tGradCurTemp;
    }

    Plato::ScalarVector gradientCurrentVel
    (const Plato::ScalarVector & aControls,
     const Plato::Primal & aVariables) const override
    {
        const auto tNumDofs = mNumVelDofsPerNode * mSpatialModel.Mesh.nverts();
        Plato::ScalarVector tGradCurVel("gradient current velocity", tNumDofs);
        Plato::blas1::fill(0.0, tGradCurVel);
        for(auto& tCriterion : mCriteria)
        {
            auto tIndex = &tCriterion - &mCriteria[0];
            const auto tWeight = mCriterionWeights[tIndex];
            const auto tNormalization = mCriterionNormalizations[tIndex];
            auto tCriterionValue = tCriterion->value(aControls, aVariables);
            auto tCriterionGrad  = tCriterion->gradientCurrentVel(aControls, aVariables);

            auto tMultiplier = ( static_cast<Plato::Scalar>(2.0) * tWeight * tCriterionValue )
                               / ( tNormalization * tNormalization );
            Plato::blas1::update(tMultiplier, tCriterionGrad, 1.0, tGradCurVel);
        }
        return tGradCurVel;
    }

private:
    void checkInputs()
    {
        if (mCriterionNames.size() != mCriterionWeights.size())
        {
            THROWERR(std::string("Dimensions mismatch.  Number of 'Functions' and 'Weights' do not match. ") +
                     "Check scalar function with name '" + mFuncName + "'.")
        }

        if(mCriterionNames.size() != mCriterionNormalizations.size())
        {
            THROWERR(std::string("Dimensions mismatch.  Number of 'Functions' and 'Normalizations' do not match. ") +
                     "Check scalar function with name '" + mFuncName + "'.")
        }
    }

    void initialize(Teuchos::ParameterList & aInputs)
    {
        if(aInputs.sublist("Criteria").isSublist(mFuncName) == false)
        {
            THROWERR(std::string("Scalar function with tag '") + mFuncName + "' is not defined in the input file.")
        }

        auto tCriteriaInputs = aInputs.sublist("Criteria").sublist(mFuncName);
        this->parseFunctions(tCriteriaInputs);
        this->parseWeights(tCriteriaInputs);
        this->parseNormalization(tCriteriaInputs);
        this->checkInputs();

        Plato::FluidMechanics::CriterionFactory<PhysicsT> tFactory;
        for(auto& tName : mCriterionNames)
        {
            auto tScalarFunction = tFactory.createCriterion(mSpatialModel, mDataMap, aInputs, tName);
            mCriteria.push_back(tScalarFunction);
        }
    }

    void parseFunctions(Teuchos::ParameterList & aInputs)
    {
        mCriterionNames = Plato::parse_array<std::string>("Functions", aInputs);
        if(mCriterionNames.empty())
        {
            THROWERR(std::string("'Functions' keyword was not defined in function block with name '") + mFuncName
                + "'. User must define 'functions' used in a 'Weighted Sum' criterion.")
        }
    }

    void parseWeights(Teuchos::ParameterList & aInputs)
    {
        mCriterionWeights = Plato::parse_array<Plato::Scalar>("Weights", aInputs);
        if(mCriterionNames.empty())
        {
            THROWERR(std::string("'Weights' keyword was not defined in function block with name '") + mFuncName
                + "'. User must define 'weights' used in a 'Weighted Sum' criterion.")
        }
    }

    void parseNormalization(Teuchos::ParameterList & aInputs)
    {
        mCriterionNormalizations = Plato::parse_array<Plato::Scalar>("Normalizations", aInputs);
        if(mCriterionNormalizations.empty())
        {
            mCriterionNormalizations.resize(mCriterionNames.size());
            std::fill(mCriterionNormalizations.begin(), mCriterionNormalizations.end(), 1.0);
        }
    }
};
// class LeastSquares

}
// namespace FluidMechanics


// todo: physics types
template<Plato::OrdinalType SpaceDim, Plato::OrdinalType NumControls = 1>
class MomentumConservation : public Plato::SimplexFluidMechanics<SpaceDim, NumControls>
{
public:
    typedef Plato::FluidMechanics::FunctionFactory FunctionFactory;
    using SimplexT = Plato::SimplexFluidMechanics<SpaceDim, NumControls>;

    static constexpr auto mNumDofsPerNode = SimplexT::mNumMomentumDofsPerNode;
    static constexpr auto mNumDofsPerCell = SimplexT::mNumNodesPerCell * mNumDofsPerNode;
};

template<Plato::OrdinalType SpaceDim, Plato::OrdinalType NumControls = 1>
class MassConservation : public Plato::SimplexFluidMechanics<SpaceDim, NumControls>
{
public:
    typedef Plato::FluidMechanics::FunctionFactory FunctionFactory;
    using SimplexT = Plato::SimplexFluidMechanics<SpaceDim, NumControls>;

    static constexpr auto mNumDofsPerNode = SimplexT::mNumMassDofsPerNode;
    static constexpr auto mNumDofsPerCell = SimplexT::mNumNodesPerCell * mNumDofsPerNode;
};

template<Plato::OrdinalType SpaceDim, Plato::OrdinalType NumControls = 1>
class EnergyConservation : public Plato::SimplexFluidMechanics<SpaceDim, NumControls>
{
public:
    typedef Plato::FluidMechanics::FunctionFactory FunctionFactory;
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
(const Plato::Variables & aStates,
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
(const Plato::Variables & aStates)
{
    auto tElemSize = aStates.vector("element size");
    auto tVelocity = aStates.vector("current velocity");
    auto tArtificialCompressibility = aStates.vector("artificial compressibility");

    auto tReynolds = aStates.scalar("reynolds");
    auto tSafetyFactor = aStates.scalar("time step safety factor");

    auto tLength = tVelocity.size();
    Plato::ScalarVector tTimeStep("time steps", tLength);
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
(const Plato::Variables & aStates)
{
    auto tTimeStep = aStates.vector("time steps");
    auto tCurrentPressure = aStates.vector("current pressure");
    auto tPreviousPressure = aStates.vector("previous pressure");
    auto tArtificialCompress = aStates.vector("artificial compressibility");
    auto tResidual = Plato::cbs::calculate_pressure_residual(tTimeStep, tCurrentPressure, tPreviousPressure, tArtificialCompress);
    auto tStoppingCriterion = Plato::blas1::dot(tResidual, tResidual);
    return tStoppingCriterion;
}

inline Plato::Scalar
calculate_semi_implicit_solve_convergence_criterion
(const Plato::Variables & aStates)
{
    std::vector<Plato::Scalar> tErrors;

    // pressure error
    auto tTimeStep = aStates.vector("time steps");
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
class FluidsProblem : public Plato::FluidMechanics::AbstractProblem
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

    using MassConservationT     = typename Plato::MassConservation<PhysicsT::mNumSpatialDims, PhysicsT::mNumControlDofsPerNode>;
    using EnergyConservationT   = typename Plato::EnergyConservation<PhysicsT::mNumSpatialDims, PhysicsT::mNumControlDofsPerNode>;
    using MomentumConservationT = typename Plato::MomentumConservation<PhysicsT::mNumSpatialDims, PhysicsT::mNumControlDofsPerNode>;
    Plato::EssentialBCs<MassConservationT>     mPressureEssentialBCs;
    Plato::EssentialBCs<MomentumConservationT> mVelocityEssentialBCs;
    Plato::EssentialBCs<EnergyConservationT>   mTemperatureEssentialBCs;

    std::shared_ptr<Plato::AbstractSolver> mVectorFieldSolver;
    std::shared_ptr<Plato::AbstractSolver> mScalarFieldSolver;

public:
    FluidsProblem
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
        Plato::Primal tPrimalVars;
        this->calculateElemCharacteristicSize(tPrimalVars);

        for (Plato::OrdinalType tStep = 1; tStep < mNumTimeSteps; tStep++)
        {
            tPrimalVars.scalar("step", tStep);
            this->setPrimalVariables(tPrimalVars);
            this->calculateStableTimeSteps(tPrimalVars);

            this->updatePredictor(aControl, tPrimalVars);
            this->updatePressure(aControl, tPrimalVars);
            this->updateVelocity(aControl, tPrimalVars);
            this->updateTemperature(aControl, tPrimalVars);

            // todo: verify BC enforcement
            this->enforceVelocityBoundaryConditions(tPrimalVars);
            this->enforcePressureBoundaryConditions(tPrimalVars);
            this->enforceTemperatureBoundaryConditions(tPrimalVars);

            if(this->checkStoppingCriteria(tPrimalVars))
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
            Plato::Primal tPrimalVars;
            this->calculateElemCharacteristicSize(tPrimalVars);

            Plato::Scalar tOutput(0);
            auto tNumTimeSteps = mVelocity.extent(0);
            for (Plato::OrdinalType tStep = 0; tStep < tNumTimeSteps; tStep++)
            {
                tPrimalVars.scalar("step", tStep);
                auto tPressure = Kokkos::subview(mPressure, tStep, Kokkos::ALL());
                auto tVelocity = Kokkos::subview(mVelocity, tStep, Kokkos::ALL());
                auto tTemperature = Kokkos::subview(mTemperature, tStep, Kokkos::ALL());
                tPrimalVars.vector("current pressure", tPressure);
                tPrimalVars.vector("current velocity", tVelocity);
                tPrimalVars.vector("current temperature", tTemperature);
                tOutput += tItr->second->value(aControl, tPrimalVars);
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
            Plato::Dual tDualVars;
            Plato::Primal tCurPrimalVars, tPrevPrimalVars;
            this->calculateElemCharacteristicSize(tCurPrimalVars);

            auto tLastStepIndex = mVelocity.extent(0) - 1;
            Plato::ScalarVector tTotalDerivative("Total Derivative", aControl.size());
            for(Plato::OrdinalType tStep = tLastStepIndex; tStep >= 0; tStep--)
            {
                tDualVars.scalar("step", tStep);
                tCurPrimalVars.scalar("step", tStep);
                tPrevPrimalVars.scalar("step", tStep + 1);

                this->setDualVariables(tDualVars);
                this->setPrimalVariables(tCurPrimalVars);
                this->setPrimalVariables(tPrevPrimalVars);

                this->calculateStableTimeSteps(tCurPrimalVars);
                this->calculateStableTimeSteps(tPrevPrimalVars);

                this->calculateVelocityAdjoint(aName, aControl, tCurPrimalVars, tPrevPrimalVars, tDualVars);
                this->calculateTemperatureAdjoint(aName, aControl, tCurPrimalVars, tPrevPrimalVars, tDualVars);
                this->calculatePressureAdjoint(aName, aControl, tCurPrimalVars, tPrevPrimalVars, tDualVars);
                this->calculatePredictorAdjoint(aControl, tCurPrimalVars, tDualVars);

                this->calculateGradientControl(aName, aControl, tCurPrimalVars, tDualVars, tTotalDerivative);
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
            Plato::Dual tDualVars;
            Plato::Primal tCurPrimalVars, tPrevPrimalVars;
            this->calculateElemCharacteristicSize(tCurPrimalVars);

            auto tLastStepIndex = mVelocity.extent(0) - 1;
            Plato::ScalarVector tTotalDerivative("Total Derivative", aControl.size());
            for(Plato::OrdinalType tStep = tLastStepIndex; tStep >= 0; tStep--)
            {
                tDualVars.scalar("step", tStep);
                tCurPrimalVars.scalar("step", tStep);
                tPrevPrimalVars.scalar("step", tStep + 1);

                this->setDualVariables(tDualVars);
                this->setPrimalVariables(tCurPrimalVars);
                this->setPrimalVariables(tPrevPrimalVars);

                this->calculateStableTimeSteps(tCurPrimalVars);
                this->calculateStableTimeSteps(tPrevPrimalVars);

                this->calculateVelocityAdjoint(aName, aControl, tCurPrimalVars, tPrevPrimalVars, tDualVars);
                this->calculateTemperatureAdjoint(aName, aControl, tCurPrimalVars, tPrevPrimalVars, tDualVars);
                this->calculatePressureAdjoint(aName, aControl, tCurPrimalVars, tPrevPrimalVars, tDualVars);
                this->calculatePredictorAdjoint(aControl, tCurPrimalVars, tDualVars);

                this->calculateGradientConfig(aName, aControl, tCurPrimalVars, tDualVars, tTotalDerivative);
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

    bool checkStoppingCriteria(const Plato::Primal & aVariables)
    {
        bool tStop = false;
        if(!mIsTransientProblem)
        {
            Plato::Scalar tCriterionValue(0.0);
            if(mIsExplicitSolve)
            {
                tCriterionValue = Plato::cbs::calculate_explicit_solve_convergence_criterion(aVariables);
            }
            else
            {
                tCriterionValue = Plato::cbs::calculate_semi_implicit_solve_convergence_criterion(aVariables);
            }

            Plato::OrdinalType tStep = aVariables.scalar("step");
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

    void calculateElemCharacteristicSize(Plato::Primal & aVariables)
    {
        Plato::ScalarVector tElementSize;
        Plato::NodeCoordinate<mNumSpatialDims> tNodeCoordinates;
        auto tNumCells = mSpatialModel.Mesh.nverts();
        auto tElemCharacteristicSize =
            Plato::cbs::calculate_element_characteristic_size<mNumSpatialDims>(tNumCells, tNodeCoordinates);
        aVariables.vector("element size", tElemCharacteristicSize);
    }

    void calculateStableTimeSteps(Plato::Primal & aVariables)
    {
        auto tArtificialCompressibility = Plato::cbs::calculate_artificial_compressibility(aVariables);
        aVariables.vector("artificial compressibility", tArtificialCompressibility);

        aVariables.scalar("time step safety factor", mTimeStepSafetyFactor);
        auto tTimeStep = Plato::cbs::calculate_stable_time_step(aVariables);
        if(mIsTransientProblem)
        {
            Plato::Scalar tMinTimeStep(0);
            Plato::blas1::min(tTimeStep, tMinTimeStep);
            Plato::blas1::fill(tMinTimeStep, tTimeStep);
            auto tCurrentTimeStepIndex = aVariables.scalar("step index");
            auto tCurrentTime = tMinTimeStep * static_cast<Plato::Scalar>(tCurrentTimeStepIndex);
            aVariables.scalar("current time", tCurrentTime);
        }
        aVariables.vector("time steps", tTimeStep);
    }

    void enforceVelocityBoundaryConditions(Plato::Primal & aVariables)
    {
        Plato::ScalarVector tBcValues;
        Plato::LocalOrdinalVector tBcDofs;
        if(mIsTransientProblem)
        {
            auto tCurrentTime = aVariables.scalar("current time");
            mVelocityEssentialBCs.get(tBcDofs, tBcValues, tCurrentTime);
        }
        else
        {
            mVelocityEssentialBCs.get(tBcDofs, tBcValues);
        }

        auto tCurrentVelocity = aVariables.vector("current velocity");
        Plato::cbs::enforce_boundary_condition(tBcDofs, tBcValues, tCurrentVelocity);

        auto tLength = tCurrentVelocity.size();
        Plato::ScalarVector tCurrentPrescribedVelocities("prescribed velocity", tLength);
        Plato::cbs::enforce_boundary_condition(tBcDofs, tBcValues, tCurrentPrescribedVelocities);
        aVariables.vector("prescribed velocity", tCurrentPrescribedVelocities);
    }

    void enforcePressureBoundaryConditions(Plato::Primal& aVariables)
    {
        Plato::ScalarVector tBcValues;
        Plato::LocalOrdinalVector tBcDofs;
        if(mIsTransientProblem)
        {
            auto tCurrentTime = aVariables.scalar("current time");
            mPressureEssentialBCs.get(tBcDofs, tBcValues, tCurrentTime);
        }
        else
        {
            mPressureEssentialBCs.get(tBcDofs, tBcValues);
        }
        auto tCurrentPressure = aVariables.vector("current pressure");
        Plato::cbs::enforce_boundary_condition(tBcDofs, tBcValues, tCurrentPressure);
    }

    void enforceTemperatureBoundaryConditions(Plato::Primal & aVariables)
    {
        Plato::ScalarVector tBcValues;
        Plato::LocalOrdinalVector tBcDofs;
        if(mIsTransientProblem)
        {
            auto tCurrentTime = aVariables.scalar("current time");
            mTemperatureEssentialBCs.get(tBcDofs, tBcValues, tCurrentTime);
        }
        else
        {
            mTemperatureEssentialBCs.get(tBcDofs, tBcValues);
        }
        auto tCurrentTemperature = aVariables.vector("current temperature");
        Plato::cbs::enforce_boundary_condition(tBcDofs, tBcValues, tCurrentTemperature);
    }

    void setDualVariables(Plato::Dual & aVariables)
    {
        if(aVariables.isVectorMapEmpty())
        {
            // FIRST BACKWARD TIME INTEGRATION STEP
            auto tTotalNumNodes = mSpatialModel.Mesh.nverts();
            std::vector<std::string> tNames =
                {"current pressure adjoint" , "current temperature adjoint",
                "previous pressure adjoint", "previous temperature adjoint"};
            for(auto& tName : tNames)
            {
                Plato::ScalarVector tView(tName, tTotalNumNodes);
                aVariables.vector(tName, tView);
            }

            auto tTotalNumDofs = mNumVelDofsPerNode * tTotalNumNodes;
            tNames = {"current velocity adjoint" , "current predictor adjoint" ,
                      "previous velocity adjoint", "previous predictor adjoint"};
            for(auto& tName : tNames)
            {
                Plato::ScalarVector tView(tName, tTotalNumDofs);
                aVariables.vector(tName, tView);
            }
        }
        else
        {
            // N-TH BACKWARD TIME INTEGRATION STEP
            std::vector<std::string> tNames =
                {"pressure adjoint", "temperature adjoint", "velocity adjoint", "predictor adjoint" };
            for(auto& tName : tNames)
            {
                auto tVector = aVariables.vector(std::string("current ") + tName);
                aVariables.vector(std::string("previous ") + tName, tVector);
            }
        }
    }



    void setPrimalVariables(Plato::Primal & aVariables)
    {
        Plato::OrdinalType tStep = aVariables.scalar("step");
        auto tCurrentVel   = Kokkos::subview(mVelocity, tStep, Kokkos::ALL());
        auto tCurrentPred  = Kokkos::subview(mPredictor, tStep, Kokkos::ALL());
        auto tCurrentTemp  = Kokkos::subview(mTemperature, tStep, Kokkos::ALL());
        auto tCurrentPress = Kokkos::subview(mPressure, tStep, Kokkos::ALL());
        aVariables.vector("current velocity", tCurrentVel);
        aVariables.vector("current pressure", tCurrentPress);
        aVariables.vector("current temperature", tCurrentTemp);
        aVariables.vector("current predictor", tCurrentPred);

        auto tPrevStep = tStep - 1;
        if (tPrevStep >= static_cast<Plato::OrdinalType>(0))
        {
            auto tPreviouVel    = Kokkos::subview(mVelocity, tPrevStep, Kokkos::ALL());
            auto tPreviousPred  = Kokkos::subview(mPredictor, tStep, Kokkos::ALL());
            auto tPreviousTemp  = Kokkos::subview(mTemperature, tPrevStep, Kokkos::ALL());
            auto tPreviousPress = Kokkos::subview(mPressure, tPrevStep, Kokkos::ALL());
            aVariables.vector("previous velocity", tPreviouVel);
            aVariables.vector("previous predictor", tPreviousPred);
            aVariables.vector("previous pressure", tPreviousPress);
            aVariables.vector("previous temperature", tPreviousTemp);
        }
        else
        {
            auto tLength = mPressure.extent(1);
            Plato::ScalarVector tPreviousPress("previous pressure", tLength);
            aVariables.vector("previous pressure", tPreviousPress);
            tLength = mTemperature.extent(1);
            Plato::ScalarVector tPreviousTemp("previous temperature", tLength);
            aVariables.vector("previous temperature", tPreviousTemp);
            tLength = mVelocity.extent(1);
            Plato::ScalarVector tPreviousVel("previous velocity", tLength);
            aVariables.vector("previous velocity", tPreviousVel);
            tLength = mPredictor.extent(1);
            Plato::ScalarVector tPreviousPred("previous predictor", tLength);
            aVariables.vector("previous previous predictor", tPreviousPred);
        }
    }

    void updateVelocity
    (const Plato::ScalarVector & aControl,
           Plato::Primal       & aVariables)
    {
        // calculate current residual and jacobian matrix
        auto tResidualVelocity = mVelocityResidual.value(aVariables);
        auto tJacobianVelocity = mVelocityResidual.gradientCurrentVel(aVariables);

        // solve velocity equation (consistent or mass lumped)
        Plato::ScalarVector tDeltaVelocity("increment", tResidualVelocity.size());
        Plato::blas1::fill(static_cast<Plato::Scalar>(0.0), tDeltaVelocity);
        mVectorFieldSolver->solve(*tJacobianVelocity, tDeltaVelocity, tResidualVelocity);

        // update velocity
        auto tCurrentVelocity  = aVariables.vector("current velocity");
        auto tPreviousVelocity = aVariables.vector("previous velocity");
        Plato::blas1::copy(tPreviousVelocity, tCurrentVelocity);
        Plato::blas1::axpy(1.0, tDeltaVelocity, tCurrentVelocity);
    }

    void updatePredictor
    (const Plato::ScalarVector & aControl,
           Plato::Primal       & aVariables)
    {
        // calculate current residual and jacobian matrix
        auto tResidualPredictor = mPredictorResidual.value(aVariables);
        auto tJacobianPredictor = mPredictorResidual.gradientPredictor(aVariables);

        // solve predictor equation (consistent or mass lumped)
        Plato::ScalarVector tDeltaPredictor("increment", tResidualPredictor.size());
        Plato::blas1::fill(static_cast<Plato::Scalar>(0.0), tDeltaPredictor);
        mVectorFieldSolver->solve(*tJacobianPredictor, tDeltaPredictor, tResidualPredictor);

        // update current predictor
        auto tCurrentPredictor  = aVariables.vector("current predictor");
        auto tPreviousPredictor = aVariables.vector("previous predictor");
        Plato::blas1::copy(tPreviousPredictor, tCurrentPredictor);
        Plato::blas1::axpy(1.0, tDeltaPredictor, tCurrentPredictor);
    }

    void updatePressure
    (const Plato::ScalarVector & aControl,
           Plato::Primal       & aVariables)
    {
        // calculate current residual and jacobian matrix
        auto tResidualPressure = mPressureResidual.value(aVariables);
        auto tJacobianPressure = mPressureResidual.gradientCurrentPress(aVariables);

        // solve mass equation (consistent or mass lumped)
        Plato::ScalarVector tDeltaPressure("increment", tResidualPressure.size());
        Plato::blas1::fill(static_cast<Plato::Scalar>(0.0), tDeltaPressure);
        mScalarFieldSolver->solve(*tJacobianPressure, tDeltaPressure, tResidualPressure);

        // update pressure
        auto tCurrentPressure = aVariables.vector("current pressure");
        auto tPreviousPressure = aVariables.vector("previous pressure");
        Plato::blas1::copy(tPreviousPressure, tCurrentPressure);
        Plato::blas1::axpy(1.0, tDeltaPressure, tCurrentPressure);
    }

    void updateTemperature
    (const Plato::ScalarVector & aControl,
           Plato::Primal       & aVariables)
    {
        // calculate current residual and jacobian matrix
        auto tResidualTemperature = mTemperatureResidual.value(aVariables);
        auto tJacobianTemperature = mTemperatureResidual.gradientCurrentTemp(aVariables);

        // solve energy equation (consistent or mass lumped)
        Plato::ScalarVector tDeltaTemperature("increment", tResidualTemperature.size());
        Plato::blas1::fill(static_cast<Plato::Scalar>(0.0), tDeltaTemperature);
        mScalarFieldSolver->solve(*tJacobianTemperature, tDeltaTemperature, tResidualTemperature);

        // update temperature
        auto tCurrentTemperature  = aVariables.vector("current temperature");
        auto tPreviousTemperature = aVariables.vector("previous temperature");
        Plato::blas1::copy(tPreviousTemperature, tCurrentTemperature);
        Plato::blas1::axpy(1.0, tDeltaTemperature, tCurrentTemperature);
    }

    void calculatePredictorAdjoint
    (const Plato::ScalarVector & aControl,
     const Plato::Primal       & aCurPrimalVars,
           Plato::Dual         & aDualVars)
    {

        auto tCurrentVelocityAdjoint = aDualVars.vector("current velocity adjoint");
        auto tGradResVelWrtPredictor = mVelocityResidual.gradientPredictor(aControl, aCurPrimalVars);
        Plato::ScalarVector tRHS("right hand side", tCurrentVelocityAdjoint.size());
        Plato::MatrixTimesVectorPlusVector(tGradResVelWrtPredictor, tCurrentVelocityAdjoint, tRHS);

        auto tCurrentPressureAdjoint = aDualVars.vector("current pressure adjoint");
        auto tGradResPressWrtPredictor = mPressureResidual.gradientPredictor(aControl, aCurPrimalVars);
        Plato::MatrixTimesVectorPlusVector(tGradResPressWrtPredictor, tCurrentPressureAdjoint, tRHS);
        Plato::blas1::scale(-1.0, tRHS);

        auto tCurrentPredictorAdjoint = aDualVars.vector("current predictor adjoint");
        Plato::blas1::fill(0.0, tCurrentPredictorAdjoint);
        auto tJacobianPredictor = mPredictorResidual.gradientPredictor(aControl, aCurPrimalVars);
        mVectorFieldSolver->solve(*tJacobianPredictor, tCurrentPredictorAdjoint, tRHS);
    }

    void calculatePressureAdjoint
    (const std::string         & aName,
     const Plato::ScalarVector & aControl,
     const Plato::Primal       & aCurPrimalVars,
     const Plato::Primal       & aPrevPrimalVars,
           Plato::Dual         & aDualVars)
    {
        auto tRHS = mCriteria[aName]->gradientCurrentPress(aControl, aCurPrimalVars);

        auto tGradResVelWrtCurPress = mVelocityResidual.gradientCurrentPress(aControl, aCurPrimalVars);
        auto tCurrentVelocityAdjoint = aDualVars.vector("current velocity adjoint");
        Plato::MatrixTimesVectorPlusVector(tGradResVelWrtCurPress, tCurrentVelocityAdjoint, tRHS);

        auto tGradResPressWrtPrevPress = mPressureResidual.gradientPreviousPress(aControl, aPrevPrimalVars);
        auto tPrevPressureAdjoint = aDualVars.vector("previous pressure adjoint");
        Plato::MatrixTimesVectorPlusVector(tGradResPressWrtPrevPress, tPrevPressureAdjoint, tRHS);

        auto tGradResVelWrtPrevPress = mVelocityResidual.gradientPreviousPress(aControl, aPrevPrimalVars);
        auto tPrevVelocityAdjoint = aDualVars.vector("previous velocity adjoint");
        Plato::MatrixTimesVectorPlusVector(tGradResVelWrtPrevPress, tPrevVelocityAdjoint, tRHS);

        auto tGradResPredWrtPrevPress = mPredictorResidual.gradientPreviousPress(aControl, aPrevPrimalVars);
        auto tPrevPredictorAdjoint = aDualVars.vector("previous predictor adjoint");
        Plato::MatrixTimesVectorPlusVector(tGradResPredWrtPrevPress, tPrevPredictorAdjoint, tRHS);
        Plato::blas1::scale(-1.0, tRHS);

        auto tCurrentPressAdjoint = aDualVars.vector("current pressure adjoint");
        Plato::blas1::fill(0.0, tCurrentPressAdjoint);
        auto tJacobianPressure = mPressureResidual.gradientCurrentPress(aControl, aCurPrimalVars);
        mScalarFieldSolver->solve(*tJacobianPressure, tCurrentPressAdjoint, tRHS);
    }

    void calculateTemperatureAdjoint
    (const std::string         & aName,
     const Plato::ScalarVector & aControl,
     const Plato::Primal       & aCurPrimalVars,
     const Plato::Primal       & aPrevPrimalVars,
           Plato::Dual         & aDualVars)
    {
        auto tRHS = mCriteria[aName]->gradientCurrentTemp(aControl, aCurPrimalVars);

        auto tGradResPredWrtPrevTemp = mPredictorResidual.gradientPreviousTemp(aControl, aPrevPrimalVars);
        auto tPrevPredictorAdjoint = aDualVars.vector("previous predictor adjoint");
        Plato::MatrixTimesVectorPlusVector(tGradResPredWrtPrevTemp, tPrevPredictorAdjoint, tRHS);

        auto tGradResTempWrtPrevTemp = mTemperatureResidual.gradientPreviousTemp(aControl, aPrevPrimalVars);
        auto tPrevTempAdjoint = aDualVars.vector("previous temperature adjoint");
        Plato::MatrixTimesVectorPlusVector(tGradResTempWrtPrevTemp, tPrevTempAdjoint, tRHS);
        Plato::blas1::scale(-1.0, tRHS);

        auto tCurrentTempAdjoint = aDualVars.vector("current temperature adjoint");
        Plato::blas1::fill(0.0, tCurrentTempAdjoint);
        auto tJacobianTemperature = mTemperatureResidual.gradientCurrentTemp(aControl, aCurPrimalVars);
        mScalarFieldSolver->solve(*tJacobianTemperature, tCurrentTempAdjoint, tRHS);
    }

    void calculateVelocityAdjoint
    (const std::string         & aName,
     const Plato::ScalarVector & aControl,
     const Plato::Primal       & aCurPrimalVars,
     const Plato::Primal       & aPrevPrimalVars,
           Plato::Dual         & aDualVars)
    {
        auto tRHS = mCriteria[aName]->gradientCurrentVel(aControl, aCurPrimalVars);

        auto tGradResPredWrtPrevVel = mPredictorResidual.gradientPreviousVel(aControl, aPrevPrimalVars);
        auto tPrevPredictorAdjoint = aDualVars.vector("previous predictor adjoint");
        Plato::MatrixTimesVectorPlusVector(tGradResPredWrtPrevVel, tPrevPredictorAdjoint, tRHS);

        auto tGradResVelWrtPrevVel = mVelocityResidual.gradientPreviousVel(aControl, aPrevPrimalVars);
        auto tPrevVelocityAdjoint = aDualVars.vector("previous velocity adjoint");
        Plato::MatrixTimesVectorPlusVector(tGradResVelWrtPrevVel, tPrevVelocityAdjoint, tRHS);

        auto tGradResPressWrtPrevVel = mPressureResidual.gradientPreviousVel(aControl, aPrevPrimalVars);
        auto tPrevPressureAdjoint = aDualVars.vector("previous pressure adjoint");
        Plato::MatrixTimesVectorPlusVector(tGradResPressWrtPrevVel, tPrevPressureAdjoint, tRHS);

        auto tGradResTempWrtPrevVel = mTemperatureResidual.gradientPreviousVel(aControl, aPrevPrimalVars);
        auto tPrevTemperatureAdjoint = aDualVars.vector("previous temperature adjoint");
        Plato::MatrixTimesVectorPlusVector(tGradResTempWrtPrevVel, tPrevTemperatureAdjoint, tRHS);
        Plato::blas1::scale(-1.0, tRHS);

        auto tCurrentVelocityAdjoint = aDualVars.vector("current velocity adjoint");
        Plato::blas1::fill(0.0, tCurrentVelocityAdjoint);
        auto tJacobianVelocity = mVelocityResidual.gradientCurrentVel(aControl, aCurPrimalVars);
        mVectorFieldSolver->solve(*tJacobianVelocity, tCurrentVelocityAdjoint, tRHS);
    }

    void calculateGradientControl
    (const std::string         & aName,
     const Plato::ScalarVector & aControl,
     const Plato::Primal       & aCurPrimalVars,
     const Plato::Dual         & aDualVars,
           Plato::ScalarVector & aTotalDerivative)
    {
        auto tGradCriterionWrtControl = mCriteria[aName]->gradientControl(aControl, aCurPrimalVars);

        auto tCurrentPredictorAdjoint = aDualVars.vector("current predictor adjoint");
        auto tGradResPredWrtControl = mPredictorResidual.gradientControl(aControl, aCurPrimalVars);
        Plato::MatrixTimesVectorPlusVector(tGradResPredWrtControl, tCurrentPredictorAdjoint, tGradCriterionWrtControl);

        auto tCurrentPressureAdjoint = aDualVars.vector("current pressure adjoint");
        auto tGradResPressWrtControl = mPressureResidual.gradientControl(aControl, aCurPrimalVars);
        Plato::MatrixTimesVectorPlusVector(tGradResPressWrtControl, tCurrentPressureAdjoint, tGradCriterionWrtControl);

        auto tCurrentTemperatureAdjoint = aDualVars.vector("current temperature adjoint");
        auto tGradResTempWrtControl = mTemperatureResidual.gradientControl(aControl, aCurPrimalVars);
        Plato::MatrixTimesVectorPlusVector(tGradResTempWrtControl, tCurrentTemperatureAdjoint, tGradCriterionWrtControl);

        auto tCurrentVelocityAdjoint = aDualVars.vector("current velocity adjoint");
        auto tGradResVelWrtControl = mVelocityResidual.gradientControl(aControl, aCurPrimalVars);
        Plato::MatrixTimesVectorPlusVector(tGradResVelWrtControl, tCurrentVelocityAdjoint, tGradCriterionWrtControl);

        Plato::blas1::axpy(1.0, tGradCriterionWrtControl, aTotalDerivative);
    }

    void calculateGradientConfig
    (const std::string         & aName,
     const Plato::ScalarVector & aControl,
     const Plato::Primal       & aCurPrimalVars,
     const Plato::Dual         & aDualVars,
           Plato::ScalarVector & aTotalDerivative)
    {
        auto tGradCriterionWrtConfig = mCriteria[aName]->gradientConfig(aControl, aCurPrimalVars);

        auto tCurrentPredictorAdjoint = aDualVars.vector("current predictor adjoint");
        auto tGradResPredWrtConfig = mPredictorResidual.gradientConfig(aControl, aCurPrimalVars);
        Plato::MatrixTimesVectorPlusVector(tGradResPredWrtConfig, tCurrentPredictorAdjoint, tGradCriterionWrtConfig);

        auto tCurrentPressureAdjoint = aDualVars.vector("current pressure adjoint");
        auto tGradResPressWrtConfig = mPressureResidual.gradientConfig(aControl, aCurPrimalVars);
        Plato::MatrixTimesVectorPlusVector(tGradResPressWrtConfig, tCurrentPressureAdjoint, tGradCriterionWrtConfig);

        auto tCurrentTemperatureAdjoint = aDualVars.vector("current temperature adjoint");
        auto tGradResTempWrtConfig = mTemperatureResidual.gradientConfig(aControl, aCurPrimalVars);
        Plato::MatrixTimesVectorPlusVector(tGradResTempWrtConfig, tCurrentTemperatureAdjoint, tGradCriterionWrtConfig);

        auto tCurrentVelocityAdjoint = aDualVars.vector("current velocity adjoint");
        auto tGradResVelWrtConfig = mVelocityResidual.gradientConfig(aControl, aCurPrimalVars);
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
            mTemperatureEssentialBCs = Plato::EssentialBCs<EnergyConservationT>(tTempBCs, mSpatialModel.MeshSets);
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
            mPressureEssentialBCs = Plato::EssentialBCs<MassConservationT>(tPressBCs, mSpatialModel.MeshSets);
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
            mVelocityEssentialBCs = Plato::EssentialBCs<MomentumConservationT>(tVelBCs, mSpatialModel.MeshSets);
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

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, EntityFaceOrdinals)
{
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);
    auto tMeshSets = PlatoUtestHelpers::get_box_mesh_sets(tMesh.operator*());

    // test: node sets
    auto tMyNodeSetOrdinals = Plato::get_entity_ordinals<Omega_h::NODE_SET>(tMeshSets, "x+");
    auto tLength = tMyNodeSetOrdinals.size();
    Plato::LocalOrdinalVector tNodeSetOrdinals("node set ordinals", tLength);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tLength), LAMBDA_EXPRESSION(const Plato::OrdinalType & aOrdinal)
    {
        tNodeSetOrdinals(aOrdinal) = tMyNodeSetOrdinals[aOrdinal];
    }, "copy");
    auto tHostNodeSetOrdinals = Kokkos::create_mirror(tNodeSetOrdinals);
    Kokkos::deep_copy(tHostNodeSetOrdinals, tNodeSetOrdinals);
    TEST_EQUALITY(2, tHostNodeSetOrdinals(0));
    TEST_EQUALITY(3, tHostNodeSetOrdinals(1));
    //Plato::omega_h::print(tMyNodeSetOrdinals, "ordinals");

    // test: side sets
    auto tMySideSetOrdinals = Plato::get_entity_ordinals<Omega_h::SIDE_SET>(tMeshSets, "x+");
    tLength = tMySideSetOrdinals.size();
    Plato::LocalOrdinalVector tSideSetOrdinals("side set ordinals", tLength);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tLength), LAMBDA_EXPRESSION(const Plato::OrdinalType & aOrdinal)
    {
        tSideSetOrdinals(aOrdinal) = tMySideSetOrdinals[aOrdinal];
    }, "copy");
    auto tHostSideSetOrdinals = Kokkos::create_mirror(tSideSetOrdinals);
    Kokkos::deep_copy(tHostSideSetOrdinals, tSideSetOrdinals);
    TEST_EQUALITY(4, tHostSideSetOrdinals(0));
    //Plato::omega_h::print(tMySideSetOrdinals, "ordinals");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, IsEntitySetDefined)
{
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);
    auto tMeshSets = PlatoUtestHelpers::get_box_mesh_sets(tMesh.operator*());
    TEST_EQUALITY(true, Plato::is_entity_set_defined<Omega_h::NODE_SET>(tMeshSets, "x+"));
    TEST_EQUALITY(false, Plato::is_entity_set_defined<Omega_h::NODE_SET>(tMeshSets, "dog"));

    TEST_EQUALITY(true, Plato::is_entity_set_defined<Omega_h::SIDE_SET>(tMeshSets, "x+"));
    TEST_EQUALITY(false, Plato::is_entity_set_defined<Omega_h::SIDE_SET>(tMeshSets, "dog"));
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, VelocityPredictorResidual)
{
    // set xml file inputs
    Teuchos::RCP<Teuchos::ParameterList> tInputs =
        Teuchos::getParametersFromXmlString(
            "<ParameterList name='Problem'>"
            "  <ParameterList name='Hyperbolic'>"
            "    <ParameterList  name='Dimensionless Properties'>"
            "      <Parameter  name='Darcy Number'   type='double'        value='1.0'/>"
            "      <Parameter  name='Prandtl Number' type='double'        value='1.0'/>"
            "      <Parameter  name='Grashof Number' type='Array(double)' value='{0.0,1.0}'/>"
            "    </ParameterList>"
            "    <ParameterList name='Momentum Conservation'>"
            "      <ParameterList name='Penalty Function'>"
            "        <Parameter name='Grashof Penalty Exponent'     type='double' value='3.0'/>"
            "        <Parameter name='Prandtl Convexity Parameter'  type='double' value='0.5'/>"
            "        <Parameter name='Brinkman Convexity Parameter' type='double' value='0.5'/>"
            "      </ParameterList>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList  name='Momentum Natural Boundary Conditions'>"
            "    <ParameterList  name='Traction Vector Boundary Condition'>"
            "      <Parameter  name='Type'   type='string'        value='Uniform'/>"
            "      <Parameter  name='Sides'  type='string'        value='x+'/>"
            "      <Parameter  name='Values' type='Array(double)' value='{0,-1.0,0}'/>"
            "    </ParameterList>"
            "  </ParameterList>"
            "</ParameterList>"
            );

    // build mesh, spatial domain, and spatial model
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);
    auto tMeshSets = PlatoUtestHelpers::get_box_mesh_sets(tMesh.operator*());
    Plato::SpatialDomain tDomain(tMesh.operator*(), tMeshSets, "box");
    tDomain.cellOrdinals("body");
    Plato::SpatialModel tModel(tMesh.operator*(), tMeshSets);
    tModel.append(tDomain);

    // set physics and evaluation type
    constexpr Plato::OrdinalType tNumSpaceDims = 2;
    using PhysicsT = Plato::IncompressibleFluids<tNumSpaceDims>::MomentumPhysicsT;

    // set control variables
    auto tNumNodes = tMesh->nverts();
    Plato::ScalarVector tControls("control", tNumNodes);
    Plato::blas1::fill(1.0, tControls);

    // set state variables
    Plato::Primal tVariables;
    auto tNumVelDofs = tNumNodes * tNumSpaceDims;
    Plato::ScalarVector tPrevVel("previous velocity", tNumVelDofs);
    auto tHostPrevVel = Kokkos::create_mirror(tPrevVel);
    tHostPrevVel(0) = 1.0; tHostPrevVel(1) = 1.1; tHostPrevVel(2) = 1.2;
    tHostPrevVel(3) = 1.3; tHostPrevVel(4) = 1.4; tHostPrevVel(5) = 1.5;
    Kokkos::deep_copy(tPrevVel, tHostPrevVel);
    tVariables.vector("previous velocity", tPrevVel);
    Plato::ScalarVector tPrevPress("previous pressure", tNumNodes);
    Plato::blas1::fill(1.0, tPrevPress);
    tVariables.vector("previous pressure", tPrevPress);
    Plato::ScalarVector tPrevTemp("previous temperature", tNumNodes);
    Plato::blas1::fill(1.0, tPrevTemp);
    tVariables.vector("previous temperature", tPrevTemp);

    Plato::ScalarVector tCurVel("current velocity", tNumVelDofs);
    tVariables.vector("current velocity", tCurVel);
    Plato::ScalarVector tCurPred("current predictor", tNumVelDofs);
    tVariables.vector("current predictor", tCurPred);
    Plato::ScalarVector tCurPress("current pressure", tNumNodes);
    tVariables.vector("current pressure", tCurPress);
    Plato::ScalarVector tCurTemp("current temperature", tNumNodes);
    tVariables.vector("current temperature", tCurTemp);

    Plato::ScalarVector tTimeSteps("time step", tNumNodes);
    Plato::blas1::fill(0.1, tTimeSteps);
    tVariables.vector("time steps", tTimeSteps);
    Plato::ScalarVector tVelBCs("prescribed velocity", tNumVelDofs);
    Plato::blas1::fill(1.0, tVelBCs);
    tVariables.vector("prescribed velocity", tVelBCs);

    // allocate vector function
    Plato::DataMap tDataMap;
    std::string tFuncName("Velocity Predictor");
    Plato::FluidMechanics::VectorFunction<PhysicsT> tVectorFunction(tFuncName, tModel, tDataMap, tInputs.operator*());

    // test vector function value
    auto tResidual = tVectorFunction.value(tControls, tVariables);

    auto tHostResidual = Kokkos::create_mirror(tResidual);
    Kokkos::deep_copy(tHostResidual, tResidual);
    std::vector<Plato::Scalar> tGold =
        {-3.383333e-01, -3.344167e-01, -1.913333e-01, -1.909167e-01, -3.294444e-01, -1.707593e-01, -2.175556e-01, -7.390741e-02};
    auto tTol = 1e-4;
    for(auto& tValue : tGold)
    {
        auto tIndex = &tValue - &tGold[0];
        TEST_FLOATING_EQUALITY(tValue,tHostResidual(tIndex),tTol);
    } 
    //Plato::print(tResidual, "residual");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, GetNumEntities)
{
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);
    auto tNumEntities = Plato::get_num_entities(Omega_h::VERT, tMesh.operator*());
    TEST_EQUALITY(4, tNumEntities);
    tNumEntities = Plato::get_num_entities(Omega_h::EDGE, tMesh.operator*());
    TEST_EQUALITY(5, tNumEntities);
    tNumEntities = Plato::get_num_entities(Omega_h::FACE, tMesh.operator*());
    TEST_EQUALITY(2, tNumEntities);
    tNumEntities = Plato::get_num_entities(Omega_h::REGION, tMesh.operator*());
    TEST_EQUALITY(2, tNumEntities);
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, FacesOnNonPrescribedBoundary)
{
    // build mesh, mesh sets, and spatial domain
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);
    auto tMeshSets = PlatoUtestHelpers::get_box_mesh_sets(tMesh.operator*());
    Plato::SpatialDomain tDomain(tMesh.operator*(), tMeshSets, "box");
    tDomain.cellOrdinals("body");

    // test 1
    std::vector<std::string> tNames = {"x-","x+","y+","y-"};
    auto tFaceOrdinalsOnBoundaryOne = Plato::entities_on_non_prescribed_boundary<Omega_h::EDGE, Omega_h::SIDE_SET>(tNames, tDomain.Mesh, tDomain.MeshSets);
    TEST_EQUALITY(0, tFaceOrdinalsOnBoundaryOne.size());

    // test 2
    tNames = {"x+","y+","y-"};
    auto tFaceOrdinalsOnBoundaryTwo = Plato::entities_on_non_prescribed_boundary<Omega_h::EDGE, Omega_h::SIDE_SET>(tNames, tDomain.Mesh, tDomain.MeshSets);
    auto tLength = tFaceOrdinalsOnBoundaryTwo.size();
    TEST_EQUALITY(1, tLength);
    Plato::ScalarVector tValuesUseCaseTwo("use case two", tLength);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tLength), LAMBDA_EXPRESSION(const Plato::OrdinalType & aOrdinal)
    {
        tValuesUseCaseTwo(aOrdinal) = tFaceOrdinalsOnBoundaryTwo[aOrdinal];
    },"data");
    auto tHostValuesUseCaseTwo = Kokkos::create_mirror(tValuesUseCaseTwo);
    Kokkos::deep_copy(tHostValuesUseCaseTwo, tValuesUseCaseTwo);
    TEST_EQUALITY(2, static_cast<Plato::OrdinalType>(tHostValuesUseCaseTwo(0)));

    // test 3
    tNames = {"y+","y-"};
    auto tFaceOrdinalsOnBoundaryThree = Plato::entities_on_non_prescribed_boundary<Omega_h::EDGE, Omega_h::SIDE_SET>(tNames, tDomain.Mesh, tDomain.MeshSets);
    tLength = tFaceOrdinalsOnBoundaryThree.size();
    TEST_EQUALITY(2, tLength);
    Plato::ScalarVector tValuesUseCaseThree("use case three", tLength);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tLength), LAMBDA_EXPRESSION(const Plato::OrdinalType & aOrdinal)
    {
        tValuesUseCaseThree(aOrdinal) = tFaceOrdinalsOnBoundaryThree[aOrdinal];
    },"data");
    auto tHostValuesUseCaseThree = Kokkos::create_mirror(tValuesUseCaseThree);
    Kokkos::deep_copy(tHostValuesUseCaseThree, tValuesUseCaseThree);
    TEST_EQUALITY(2, static_cast<Plato::OrdinalType>(tHostValuesUseCaseThree(0)));
    TEST_EQUALITY(4, static_cast<Plato::OrdinalType>(tHostValuesUseCaseThree(1)));

    // test 4
    tNames = {"y-"};
    auto tFaceOrdinalsOnBoundaryFour = Plato::entities_on_non_prescribed_boundary<Omega_h::EDGE, Omega_h::SIDE_SET>(tNames, tDomain.Mesh, tDomain.MeshSets);
    tLength = tFaceOrdinalsOnBoundaryFour.size();
    TEST_EQUALITY(3, tLength);
    Plato::ScalarVector tValuesUseCaseFour("use case four", tLength);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tLength), LAMBDA_EXPRESSION(const Plato::OrdinalType & aOrdinal)
    {
        tValuesUseCaseFour(aOrdinal) = tFaceOrdinalsOnBoundaryFour[aOrdinal];
    },"data");
    auto tHostValuesUseCaseFour = Kokkos::create_mirror(tValuesUseCaseFour);
    Kokkos::deep_copy(tHostValuesUseCaseFour, tValuesUseCaseFour);
    TEST_EQUALITY(2, static_cast<Plato::OrdinalType>(tHostValuesUseCaseFour(0)));
    TEST_EQUALITY(3, static_cast<Plato::OrdinalType>(tHostValuesUseCaseFour(1)));
    TEST_EQUALITY(4, static_cast<Plato::OrdinalType>(tHostValuesUseCaseFour(2)));

    // test 5
    tNames = {};
    auto tFaceOrdinalsOnBoundaryFive = Plato::entities_on_non_prescribed_boundary<Omega_h::EDGE, Omega_h::SIDE_SET>(tNames, tDomain.Mesh, tDomain.MeshSets);
    tLength = tFaceOrdinalsOnBoundaryFive.size();
    TEST_EQUALITY(4, tLength);
    Plato::ScalarVector tValuesUseCaseFive("use case five", tLength);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tLength), LAMBDA_EXPRESSION(const Plato::OrdinalType & aOrdinal)
    {
        tValuesUseCaseFive(aOrdinal) = tFaceOrdinalsOnBoundaryFive[aOrdinal];
    },"data");
    auto tHostValuesUseCaseFive = Kokkos::create_mirror(tValuesUseCaseFive);
    Kokkos::deep_copy(tHostValuesUseCaseFive, tValuesUseCaseFive);
    TEST_EQUALITY(0, static_cast<Plato::OrdinalType>(tHostValuesUseCaseFive(0)));
    TEST_EQUALITY(2, static_cast<Plato::OrdinalType>(tHostValuesUseCaseFive(1)));
    TEST_EQUALITY(3, static_cast<Plato::OrdinalType>(tHostValuesUseCaseFive(2)));
    TEST_EQUALITY(4, static_cast<Plato::OrdinalType>(tHostValuesUseCaseFive(3)));
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, DeviatoricSurfaceForces)
{
    // set inputs
    Teuchos::RCP<Teuchos::ParameterList> tInputs =
        Teuchos::getParametersFromXmlString(
            "<ParameterList name='Problem'>"
            "  <ParameterList name='Hyperbolic'>"
            "    <ParameterList  name='Dimensionless Properties'>"
            "      <Parameter  name='Prandtl Number' type='double' value='1.0'/>"
            "    </ParameterList>"
            "    <ParameterList name='Momentum Conservation'>"
            "      <ParameterList name='Penalty Function'>"
            "        <Parameter name='Prandtl Convexity Parameter' type='double' value='0.5'/>"
            "      </ParameterList>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList  name='Momentum Natural Boundary Conditions'>"
            "    <ParameterList  name='Traction Vector Boundary Condition'>"
            "      <Parameter  name='Type'   type='string'        value='Uniform'/>"
            "      <Parameter  name='Sides'  type='string'        value='x+'/>"
            "      <Parameter  name='Values' type='Array(double)' value='{0,-1.0}'/>"
            "    </ParameterList>"
            "  </ParameterList>"
            "</ParameterList>"
            );

    // set physics and evaluation type
    constexpr Plato::OrdinalType tNumSpaceDims = 2;
    using PhysicsT = Plato::MomentumConservation<tNumSpaceDims>;
    using ResidualEvalT = Plato::FluidMechanics::Evaluation<PhysicsT::SimplexT>::Residual;

    // build mesh, mesh sets, and spatial domain
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);
    auto tMeshSets = PlatoUtestHelpers::get_box_mesh_sets(tMesh.operator*());
    Plato::SpatialDomain tDomain(tMesh.operator*(), tMeshSets, "box");
    tDomain.cellOrdinals("body");

    // set workset
    Plato::WorkSets tWorkSets;
    auto tNumCells = tMesh->nelems();
    constexpr Plato::OrdinalType tNumNodesPerCell = tNumSpaceDims + 1;
    Plato::NodeCoordinate<tNumSpaceDims> tNodeCoordinate( (&tMesh.operator*()) );

    using ConfigT = ResidualEvalT::ConfigScalarType;
    auto tConfig = std::make_shared< Plato::MetaData< Plato::ScalarArray3DT<ConfigT> > >
        ( Plato::ScalarArray3DT<ConfigT>("configuration", tNumCells, tNumNodesPerCell, tNumSpaceDims) );
    Plato::workset_config_scalar<tNumSpaceDims, tNumNodesPerCell>(tMesh->nelems(), tNodeCoordinate, tConfig->mData);
    tWorkSets.set("configuration", tConfig);

    using PrevVelT = ResidualEvalT::PreviousMomentumScalarType;
    auto tPrevVel = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<PrevVelT> > >
        ( Plato::ScalarMultiVectorT<PrevVelT>("previous velocity", tNumCells, PhysicsT::mNumMomentumDofsPerCell) );
    auto tHostVelocity = Kokkos::create_mirror(tPrevVel->mData);
    tHostVelocity(0, 0) = 0.12; tHostVelocity(1, 0) = 0.22;
    tHostVelocity(0, 1) = 0.41; tHostVelocity(1, 1) = 0.47;
    tHostVelocity(0, 2) = 0.25; tHostVelocity(1, 2) = 0.86;
    tHostVelocity(0, 3) = 0.15; tHostVelocity(1, 3) = 0.57;
    tHostVelocity(0, 4) = 0.12; tHostVelocity(1, 4) = 0.18;
    tHostVelocity(0, 5) = 0.43; tHostVelocity(1, 5) = 0.11;
    Kokkos::deep_copy(tPrevVel->mData, tHostVelocity);
    tWorkSets.set("previous velocity", tPrevVel);

    using ControlT = ResidualEvalT::ControlScalarType;
    auto tControl = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<ControlT> > >
        ( Plato::ScalarMultiVectorT<ControlT>("control", tNumCells, PhysicsT::mNumControlDofsPerCell) );
    Plato::blas2::fill(0.5, tControl->mData);
    tWorkSets.set("control", tControl);

    // build criterion
    Plato::ScalarMultiVectorT<ResidualEvalT::ResultScalarType>
        tResult("result", tNumCells, PhysicsT::mNumMomentumDofsPerCell);
    Plato::FluidMechanics::DeviatoricSurfaceForces<PhysicsT, ResidualEvalT>
        tDeviatoricSurfaceForces(tDomain, tInputs.operator*());

    // test function
    tDeviatoricSurfaceForces(tWorkSets, tResult);
    auto tHostResult = Kokkos::create_mirror(tResult);
    Kokkos::deep_copy(tHostResult, tResult);

    auto tTol = 1e-4;
    std::vector<std::vector<Plato::Scalar>> tGold = {{0.117,-0.168,0.117,-0.168,0,0},{0.384,-0.174,0.384,-0.174,0,0}};
    for (Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++)
    {
        for (Plato::OrdinalType tDof = 0; tDof < PhysicsT::mNumMomentumDofsPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(tGold[tCell][tDof], tHostResult(tCell, tDof), tTol);
            //printf("Results(Cell=%d,Dof=%d)=%f\n", tCell, tDof, tHostResult(tCell, tDof));
        }
    }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, DeviatoricSurfaceForces_Zeros)
{
    // set inputs
    Teuchos::RCP<Teuchos::ParameterList> tInputs =
        Teuchos::getParametersFromXmlString(
            "<ParameterList name='Problem'>"
            "  <ParameterList name='Hyperbolic'>"
            "    <ParameterList  name='Dimensionless Properties'>"
            "      <Parameter  name='Prandtl Number' type='double' value='1.0'/>"
            "    </ParameterList>"
            "    <ParameterList name='Momentum Conservation'>"
            "      <ParameterList name='Penalty Function'>"
            "        <Parameter name='Prandtl Convexity Parameter' type='double' value='0.5'/>"
            "      </ParameterList>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList  name='Momentum Natural Boundary Conditions'>"
            "    <ParameterList  name='Traction Vector Boundary Condition - X+'>"
            "      <Parameter  name='Sides'  type='string'        value='x+'/>"
            "      <Parameter  name='Values' type='Array(double)' value='{0,-1}'/>"
            "    </ParameterList>"
            "    <ParameterList  name='Traction Vector Boundary Condition - X-'>"
            "      <Parameter  name='Sides'  type='string'        value='x-'/>"
            "      <Parameter  name='Values' type='Array(double)' value='{0,-1}'/>"
            "    </ParameterList>"
            "    <ParameterList  name='Traction Vector Boundary Condition - Y-'>"
            "      <Parameter  name='Sides'  type='string'        value='y-'/>"
            "      <Parameter  name='Values' type='Array(double)' value='{0,-1}'/>"
            "    </ParameterList>"
            "    <ParameterList  name='Traction Vector Boundary Condition - Y+'>"
            "      <Parameter  name='Sides'  type='string'        value='y+'/>"
            "      <Parameter  name='Values' type='Array(double)' value='{0,-1}'/>"
            "    </ParameterList>"
            "  </ParameterList>"
            "</ParameterList>"
            );

    // set physics and evaluation type
    constexpr Plato::OrdinalType tNumSpaceDims = 2;
    using PhysicsT = Plato::MomentumConservation<tNumSpaceDims>;
    using ResidualEvalT = Plato::FluidMechanics::Evaluation<PhysicsT::SimplexT>::Residual;

    // build mesh, mesh sets, and spatial domain
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);
    auto tMeshSets = PlatoUtestHelpers::get_box_mesh_sets(tMesh.operator*());
    Plato::SpatialDomain tDomain(tMesh.operator*(), tMeshSets, "box");
    tDomain.cellOrdinals("body");

    // set workset
    Plato::WorkSets tWorkSets;
    auto tNumCells = tMesh->nelems();
    constexpr Plato::OrdinalType tNumNodesPerCell = tNumSpaceDims + 1;
    Plato::NodeCoordinate<tNumSpaceDims> tNodeCoordinate( (&tMesh.operator*()) );

    using ConfigT = ResidualEvalT::ConfigScalarType;
    auto tConfig = std::make_shared< Plato::MetaData< Plato::ScalarArray3DT<ConfigT> > >
        ( Plato::ScalarArray3DT<ConfigT>("configuration", tNumCells, tNumNodesPerCell, tNumSpaceDims) );
    Plato::workset_config_scalar<tNumSpaceDims, tNumNodesPerCell>(tMesh->nelems(), tNodeCoordinate, tConfig->mData);
    tWorkSets.set("configuration", tConfig);

    using PrevVelT = ResidualEvalT::PreviousMomentumScalarType;
    auto tPrevVel = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<PrevVelT> > >
        ( Plato::ScalarMultiVectorT<PrevVelT>("previous velocity", tNumCells, PhysicsT::mNumMomentumDofsPerCell) );
    auto tHostVelocity = Kokkos::create_mirror(tPrevVel->mData);
    tHostVelocity(0, 0) = 0.12; tHostVelocity(1, 0) = 0.22;
    tHostVelocity(0, 1) = 0.41; tHostVelocity(1, 1) = 0.47;
    tHostVelocity(0, 2) = 0.25; tHostVelocity(1, 2) = 0.86;
    tHostVelocity(0, 3) = 0.15; tHostVelocity(1, 3) = 0.57;
    tHostVelocity(0, 4) = 0.12; tHostVelocity(1, 4) = 0.18;
    tHostVelocity(0, 5) = 0.43; tHostVelocity(1, 5) = 0.11;
    Kokkos::deep_copy(tPrevVel->mData, tHostVelocity);
    tWorkSets.set("previous velocity", tPrevVel);

    using ControlT = ResidualEvalT::ControlScalarType;
    auto tControl = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<ControlT> > >
        ( Plato::ScalarMultiVectorT<ControlT>("control", tNumCells, PhysicsT::mNumControlDofsPerCell) );
    Plato::blas2::fill(0.5, tControl->mData);
    tWorkSets.set("control", tControl);

    // build criterion
    Plato::ScalarMultiVectorT<ResidualEvalT::ResultScalarType>
        tResult("result", tNumCells, PhysicsT::mNumMomentumDofsPerCell);
    Plato::FluidMechanics::DeviatoricSurfaceForces<PhysicsT, ResidualEvalT>
        tDeviatoricSurfaceForces(tDomain, tInputs.operator*());

    // test function
    tDeviatoricSurfaceForces(tWorkSets, tResult);
    auto tHostResult = Kokkos::create_mirror(tResult);
    Kokkos::deep_copy(tHostResult, tResult);

    auto tTol = 1e-4;
    for (Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++)
    {
        for (Plato::OrdinalType tDof = 0; tDof < PhysicsT::mNumMomentumDofsPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(0.0, tHostResult(tCell, tDof), tTol);
            //printf("Results(Cell=%d,Dof=%d)=%f\n", tCell, tDof, tHostResult(tCell, tDof));
        }
    }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PressureSurfaceForces)
{
    // set physics and evaluation type
    constexpr Plato::OrdinalType tNumSpaceDims = 2;
    using PhysicsT = Plato::MomentumConservation<tNumSpaceDims>;
    using ResidualEvalT = Plato::FluidMechanics::Evaluation<PhysicsT::SimplexT>::Residual;

    // build mesh, mesh sets, and spatial domain
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);
    auto tMeshSets = PlatoUtestHelpers::get_box_mesh_sets(tMesh.operator*());
    Plato::SpatialDomain tDomain(tMesh.operator*(), tMeshSets, "box");
    tDomain.cellOrdinals("body");

    // set workset
    Plato::WorkSets tWorkSets;
    auto tNumCells = tMesh->nelems();
    constexpr Plato::OrdinalType tNumNodesPerCell = 3;
    Plato::NodeCoordinate<tNumSpaceDims> tNodeCoordinate( (&tMesh.operator*()) );

    using ConfigT = ResidualEvalT::ConfigScalarType;
    auto tConfig = std::make_shared< Plato::MetaData< Plato::ScalarArray3DT<ConfigT> > >
        ( Plato::ScalarArray3DT<ConfigT>("configuration", tNumCells, tNumNodesPerCell, tNumSpaceDims) );
    Plato::workset_config_scalar<tNumSpaceDims, tNumNodesPerCell>(tMesh->nelems(), tNodeCoordinate, tConfig->mData);
    tWorkSets.set("configuration", tConfig);

    using PrevPressT = ResidualEvalT::PreviousMassScalarType;
    auto tPrevVel = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<PrevPressT> > >
        ( Plato::ScalarMultiVectorT<PrevPressT>("previous pressure", tNumCells, PhysicsT::mNumMassDofsPerCell) );
    Plato::blas2::fill(1, tPrevVel->mData);
    tWorkSets.set("previous pressure", tPrevVel);

    // build criterion
    Plato::ScalarMultiVectorT<ResidualEvalT::ResultScalarType>
        tResult("result", tNumCells, PhysicsT::mNumMomentumDofsPerCell);
    Plato::FluidMechanics::PressureSurfaceForces<PhysicsT, ResidualEvalT>
        tPressureSurfaceForces(tDomain, "x+");

    // test function
    tPressureSurfaceForces(tWorkSets, tResult);
    auto tHostResult = Kokkos::create_mirror(tResult);
    Kokkos::deep_copy(tHostResult, tResult);
    
    auto tTol = 1e-4;
    std::vector<std::vector<Plato::Scalar>> tGold = {{0,0,0.5,0,0.5,0},{0,0,0,0,0,0}};
    for (Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++)
    {
        for (Plato::OrdinalType tDof = 0; tDof < PhysicsT::mNumMomentumDofsPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(tGold[tCell][tDof], tHostResult(tCell, tDof), tTol);
            //printf("Results(Cell=%d,Dof=%d)=%f\n", tCell, tDof, tHostResult(tCell, tDof));
        }
    }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, StrainRate)
{
    constexpr Plato::OrdinalType tNumSpaceDims = 2;
    constexpr Plato::OrdinalType tNumNodesPerCell = 3;
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);
    TEST_EQUALITY(2, tMesh->nelems());

    auto const tNumCells = tMesh->nelems();
    Plato::NodeCoordinate<tNumSpaceDims> tNodeCoordinate( (&tMesh.operator*()) );
    Plato::ScalarArray3D tConfig("configuration", tNumCells, tNumNodesPerCell, tNumSpaceDims);
    Plato::workset_config_scalar<tNumSpaceDims,tNumNodesPerCell>(tMesh->nelems(), tNodeCoordinate, tConfig);

    Plato::ScalarVector tVolume("volume", tNumCells);
    Plato::ScalarArray3D tGradient("gradient", tNumCells, tNumNodesPerCell, tNumSpaceDims);
    Plato::ScalarArray3D tStrainRate("strain rate", tNumCells, tNumNodesPerCell, tNumSpaceDims);
    Plato::ComputeGradientWorkset<tNumSpaceDims> tComputeGradient;

    auto tNumDofsPerCell = tNumSpaceDims * tNumNodesPerCell;
    Plato::ScalarMultiVector tVelocity("velocity", tNumCells, tNumDofsPerCell);
    auto tHostVelocity = Kokkos::create_mirror(tVelocity);
    tHostVelocity(0, 0) = 0.12; tHostVelocity(1, 0) = 0.22;
    tHostVelocity(0, 1) = 0.41; tHostVelocity(1, 1) = 0.47;
    tHostVelocity(0, 2) = 0.25; tHostVelocity(1, 2) = 0.86;
    tHostVelocity(0, 3) = 0.15; tHostVelocity(1, 3) = 0.57;
    tHostVelocity(0, 4) = 0.12; tHostVelocity(1, 4) = 0.18;
    tHostVelocity(0, 5) = 0.43; tHostVelocity(1, 5) = 0.11;
    Kokkos::deep_copy(tVelocity, tHostVelocity);

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        tComputeGradient(aCellOrdinal, tGradient, tConfig, tVolume);
        Plato::FluidMechanics::strain_rate<tNumNodesPerCell, tNumSpaceDims>(aCellOrdinal, tVelocity, tGradient, tStrainRate);
    }, "strain_rate unit test");

    auto tTol = 1e-6;
    auto tHostStrainRate = Kokkos::create_mirror(tStrainRate);
    Kokkos::deep_copy(tHostStrainRate, tStrainRate);
    // cell 1
    TEST_FLOATING_EQUALITY(0.13,   tHostStrainRate(0, 0, 0), tTol);
    TEST_FLOATING_EQUALITY(-0.195, tHostStrainRate(0, 0, 1), tTol);
    TEST_FLOATING_EQUALITY(-0.195, tHostStrainRate(0, 1, 0), tTol);
    TEST_FLOATING_EQUALITY(0.28,   tHostStrainRate(0, 1, 1), tTol);
    // cell 2
    TEST_FLOATING_EQUALITY(-0.64, tHostStrainRate(1, 0, 0), tTol);
    TEST_FLOATING_EQUALITY(0.29,  tHostStrainRate(1, 0, 1), tTol);
    TEST_FLOATING_EQUALITY(0.29,  tHostStrainRate(1, 1, 0), tTol);
    TEST_FLOATING_EQUALITY(0.46,  tHostStrainRate(1, 1, 1), tTol);
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, BLAS1_DeviceScale)
{
    constexpr Plato::OrdinalType tNumCells = 2;
    constexpr Plato::OrdinalType tNumSpaceDims = 2;
    Plato::ScalarMultiVector tInput("input", tNumCells, tNumSpaceDims);
    Plato::blas2::fill(1.0, tInput);
    Plato::ScalarMultiVector tOutput("output", tNumCells, tNumSpaceDims);

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        Plato::blas1::scale<tNumSpaceDims>(aCellOrdinal, 4.0, tInput, tOutput);
    }, "device blas1::scale");

    auto tTol = 1e-6;
    auto tHostOutput = Kokkos::create_mirror(tOutput);
    Kokkos::deep_copy(tHostOutput, tOutput);
    for (Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++)
    {
        for (Plato::OrdinalType tDim = 0; tDim < tNumSpaceDims; tDim++)
        {
            TEST_FLOATING_EQUALITY(4.0, tHostOutput(tCell, tDim), tTol);
        }
    }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, BLAS1_Dot)
{
    constexpr Plato::OrdinalType tNumCells = 2;
    constexpr Plato::OrdinalType tNumSpaceDims = 2;
    Plato::ScalarMultiVector tInputA("input", tNumCells, tNumSpaceDims);
    Plato::blas2::fill(1.0, tInputA);
    Plato::ScalarMultiVector tInputB("input", tNumCells, tNumSpaceDims);
    Plato::blas2::fill(4.0, tInputB);
    Plato::ScalarVector tOutput("output", tNumCells, tNumSpaceDims, tNumSpaceDims);

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        Plato::blas1::dot<tNumSpaceDims>(aCellOrdinal, tInputA, tInputB, tOutput);
    }, "device blas1::dot");

    auto tTol = 1e-6;
    auto tHostOutput = Kokkos::create_mirror(tOutput);
    Kokkos::deep_copy(tHostOutput, tOutput);
    for (Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++)
    {
        TEST_FLOATING_EQUALITY(8.0, tHostOutput(tCell), tTol);
    }
}


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, BLAS2_DeviceScale)
{
    constexpr Plato::OrdinalType tNumCells = 2;
    constexpr Plato::OrdinalType tNumSpaceDims = 2;
    Plato::ScalarArray3D tInput("input", tNumCells, tNumSpaceDims, tNumSpaceDims);
    Plato::blas3::fill<tNumSpaceDims, tNumSpaceDims>(tNumCells, 1.0, tInput);
    Plato::ScalarArray3D tOutput("output", tNumCells, tNumSpaceDims, tNumSpaceDims);

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        Plato::blas2::scale<tNumSpaceDims, tNumSpaceDims>(aCellOrdinal, 4.0, tInput, tOutput);
    }, "device blas2::scale");

    auto tTol = 1e-6;
    auto tHostOutput = Kokkos::create_mirror(tOutput);
    Kokkos::deep_copy(tHostOutput, tOutput);
    for (Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++)
    {
        for (Plato::OrdinalType tDimI = 0; tDimI < tNumSpaceDims; tDimI++)
        {
            for (Plato::OrdinalType tDimJ = 0; tDimJ < tNumSpaceDims; tDimJ++)
            {
                TEST_FLOATING_EQUALITY(4.0, tHostOutput(tCell, tDimI, tDimJ), tTol);
            }
        }
    }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, BLAS2_Dot)
{
    constexpr Plato::OrdinalType tNumCells = 2;
    constexpr Plato::OrdinalType tNumSpaceDims = 2;
    Plato::ScalarArray3D tInputA("input", tNumCells, tNumSpaceDims, tNumSpaceDims);
    Plato::blas3::fill<tNumSpaceDims, tNumSpaceDims>(tNumCells, 1.0, tInputA);
    Plato::ScalarArray3D tInputB("input", tNumCells, tNumSpaceDims, tNumSpaceDims);
    Plato::blas3::fill<tNumSpaceDims, tNumSpaceDims>(tNumCells, 4.0, tInputB);
    Plato::ScalarVector tOutput("output", tNumCells, tNumSpaceDims, tNumSpaceDims);

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        Plato::blas2::dot<tNumSpaceDims, tNumSpaceDims>(aCellOrdinal, tInputA, tInputB, tOutput);
    }, "device blas2::dot");

    auto tTol = 1e-6;
    auto tHostOutput = Kokkos::create_mirror(tOutput);
    Kokkos::deep_copy(tHostOutput, tOutput);
    for (Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++)
    {
        TEST_FLOATING_EQUALITY(16.0, tHostOutput(tCell), tTol);
    }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, BrinkmanPenalization)
{
    constexpr Plato::OrdinalType tNumCells = 2;
    constexpr Plato::OrdinalType tNumNodesPerCell = 4;
    Plato::Scalar tPhysicalNum = 1.0;
    Plato::Scalar tConvexityParam = 0.5;
    Plato::ScalarVector tOutput("output", tNumCells);
    Plato::ScalarMultiVector tControlWS("control", tNumCells, tNumNodesPerCell);
    Plato::blas2::fill(0.5, tControlWS);

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        tOutput(aCellOrdinal) =
            Plato::FluidMechanics::brinkman_penalization<tNumNodesPerCell>(aCellOrdinal, tPhysicalNum, tConvexityParam, tControlWS);
    }, "brinkman_penalization unit test");

    auto tTol = 1e-6;
    auto tHostOutput = Kokkos::create_mirror(tOutput);
    Kokkos::deep_copy(tHostOutput, tOutput);
    for(Plato::OrdinalType tIndex = 0; tIndex < tOutput.size(); tIndex++)
    {
        TEST_FLOATING_EQUALITY(0.4, tHostOutput(tIndex), tTol);
    }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, RampPenalization)
{
    constexpr Plato::OrdinalType tNumCells = 2;
    constexpr Plato::OrdinalType tNumNodesPerCell = 4;
    Plato::Scalar tPhysicalNum = 1.0;
    Plato::Scalar tConvexityParam = 0.5;
    Plato::ScalarVector tOutput("output", tNumCells);
    Plato::ScalarMultiVector tControlWS("control", tNumCells, tNumNodesPerCell);
    Plato::blas2::fill(0.5, tControlWS);

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        tOutput(aCellOrdinal) =
            Plato::FluidMechanics::ramp_penalization<tNumNodesPerCell>(aCellOrdinal, tPhysicalNum, tConvexityParam, tControlWS);
    }, "ramp_penalization unit test");

    auto tTol = 1e-6;
    auto tHostOutput = Kokkos::create_mirror(tOutput);
    Kokkos::deep_copy(tHostOutput, tOutput);
    for(Plato::OrdinalType tIndex = 0; tIndex < tOutput.size(); tIndex++)
    {
        TEST_FLOATING_EQUALITY(0.6, tHostOutput(tIndex), tTol);
    }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, InternalDissipationEnergyIncompressible_Value)
{
    // set inputs
    Teuchos::RCP<Teuchos::ParameterList> tInputs =
        Teuchos::getParametersFromXmlString(
            "<ParameterList name='Problem'>"
            "  <ParameterList  name='Criteria'>"
            "    <Parameter  name='Type'    type='string'    value='Scalar Function'/>"
            "    <ParameterList name='My Criteria'>"
            "      <Parameter  name='Flow'                 type='string'    value='Incompressible'/>"
            "      <Parameter  name='Type'                 type='string'    value='Scalar Function'/>"
            "      <Parameter  name='Scalar Function Type' type='string'    value='Internal Dissipation Energy'/>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList  name='Dimensionless Properties'>"
            "    <Parameter  name='Darcy Number'    type='double'    value='1.0'/>"
            "    <Parameter  name='Prandtl Number' type='double'    value='1.0'/>"
            "  </ParameterList>"
            "</ParameterList>"
            );

    // build mesh, spatial domain, and spatial model
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);
    auto tMeshSets = PlatoUtestHelpers::get_box_mesh_sets(tMesh.operator*());
    Plato::SpatialDomain tDomain(tMesh.operator*(), tMeshSets, "box");
    tDomain.cellOrdinals("body");
    Plato::SpatialModel tModel(tMesh.operator*(), tMeshSets);
    tModel.append(tDomain);

    // set current state
    Plato::Variables tPrimal;
    auto tNumCells = tMesh->nelems();
    auto tNumNodes = tMesh->nverts();
    auto tNumVelDofs = tNumNodes * tMesh->dim();
    Plato::ScalarVector tControl("controls", tNumNodes);
    Plato::blas1::fill(0.5, tControl);
    Plato::ScalarVector tCurVel("current velocity", tNumVelDofs);
    Plato::blas1::fill(1.0, tCurVel);
    tPrimal.vector("current velocity", tCurVel);
    Plato::ScalarVector tCurPress("current pressure", tNumNodes);
    Plato::blas1::fill(0.1, tCurPress);
    tPrimal.vector("current pressure", tCurPress);
    Plato::ScalarVector tCurTemp("current temperature", tNumNodes);
    Plato::blas1::fill(1.5, tCurTemp);
    tPrimal.vector("current temperature", tCurTemp);
    Plato::ScalarVector tTimeSteps("time steps", tNumNodes);
    Plato::blas1::fill(0.01, tTimeSteps);
    tPrimal.vector("time steps", tTimeSteps);

    // set physics type
    constexpr Plato::OrdinalType tNumSpaceDim = 2;
    using PhysicsT = Plato::IncompressibleFluids<tNumSpaceDim>;

    // build criterion
    Plato::DataMap tDataMap;
    std::string tFuncName("My Criteria");
    Plato::FluidMechanics::PhysicsScalarFunction<PhysicsT>
        tCriterion(tModel, tDataMap, tInputs.operator*(), tFuncName);
    TEST_EQUALITY("My Criteria", tCriterion.name());

    // test criterion value
    auto tTol = 1e-4;
    auto tValue = tCriterion.value(tControl, tPrimal);
    TEST_FLOATING_EQUALITY(0.222222, tValue, tTol);
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, InternalDissipationEnergyIncompressible_GradControl)
{
    // set inputs
    Teuchos::RCP<Teuchos::ParameterList> tInputs =
        Teuchos::getParametersFromXmlString(
            "<ParameterList name='Problem'>"
            "  <ParameterList  name='Criteria'>"
            "    <Parameter  name='Type'    type='string'    value='Scalar Function'/>"
            "    <ParameterList name='My Criteria'>"
            "      <Parameter  name='Flow'                 type='string'    value='Incompressible'/>"
            "      <Parameter  name='Type'                 type='string'    value='Scalar Function'/>"
            "      <Parameter  name='Scalar Function Type' type='string'    value='Internal Dissipation Energy'/>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList  name='Dimensionless Properties'>"
            "    <Parameter  name='Darcy Number'    type='double'    value='1.0'/>"
            "    <Parameter  name='Prandtl Number'  type='double'    value='1.0'/>"
            "  </ParameterList>"
            "</ParameterList>"
            );

    // build mesh, spatial domain, and spatial model
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);
    auto tMeshSets = PlatoUtestHelpers::get_box_mesh_sets(tMesh.operator*());
    Plato::SpatialDomain tDomain(tMesh.operator*(), tMeshSets, "box");
    tDomain.cellOrdinals("body");
    Plato::SpatialModel tModel(tMesh.operator*(), tMeshSets);
    tModel.append(tDomain);

    // set current state
    Plato::Variables tPrimal;
    auto tNumCells = tMesh->nelems();
    auto tNumNodes = tMesh->nverts();
    auto tNumVelDofs = tNumNodes * tMesh->dim();
    Plato::ScalarVector tControl("controls", tNumNodes);
    Plato::blas1::fill(0.5, tControl);
    Plato::ScalarVector tCurVel("current velocity", tNumVelDofs);
    auto tHostCurVel = Kokkos::create_mirror(tCurVel);
    tHostCurVel(0, 0) = 0.12; tHostCurVel(1, 0) = 0.22;
    tHostCurVel(0, 1) = 0.41; tHostCurVel(1, 1) = 0.47;
    tHostCurVel(0, 2) = 0.25; tHostCurVel(1, 2) = 0.86;
    tHostCurVel(0, 3) = 0.15; tHostCurVel(1, 3) = 0.57;
    tHostCurVel(0, 4) = 0.12; tHostCurVel(1, 4) = 0.18;
    tHostCurVel(0, 5) = 0.43; tHostCurVel(1, 5) = 0.11;
    Kokkos::deep_copy(tCurVel, tHostCurVel);
    tPrimal.vector("current velocity", tCurVel);
    Plato::ScalarVector tCurPress("current pressure", tNumNodes);
    Plato::blas1::fill(0.1, tCurPress);
    tPrimal.vector("current pressure", tCurPress);
    Plato::ScalarVector tCurTemp("current temperature", tNumNodes);
    Plato::blas1::fill(1.5, tCurTemp);
    tPrimal.vector("current temperature", tCurTemp);
    Plato::ScalarVector tTimeSteps("time steps", tNumNodes);
    Plato::blas1::fill(0.01, tTimeSteps);
    tPrimal.vector("time steps", tTimeSteps);

    // set physics type
    constexpr Plato::OrdinalType tNumSpaceDim = 2;
    using PhysicsT = Plato::IncompressibleFluids<tNumSpaceDim>;

    // build criterion
    Plato::DataMap tDataMap;
    std::string tFuncName("My Criteria");
    Plato::FluidMechanics::PhysicsScalarFunction<PhysicsT>
        tCriterion(tModel, tDataMap, tInputs.operator*(), tFuncName);
    TEST_EQUALITY("My Criteria", tCriterion.name());

    // test criterion value
    auto tTol = 1e-4;
    auto tGradient = tCriterion.gradientControl(tControl, tPrimal);
    auto tHostGradient = Kokkos::create_mirror(tGradient);
    Kokkos::deep_copy(tHostGradient, tGradient);

    std::vector<Plato::Scalar> tGold = {-0.154798, -0.053530, -0.154798, -0.101269};
    for(Plato::OrdinalType tNode = 0; tNode < tNumNodes; tNode++)
    {
        TEST_FLOATING_EQUALITY(tGold[tNode], tHostGradient(tNode), tTol);
        //printf("Results(Node=%d)=%f\n", tNode, tHostGradient(tNode));
    }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, InternalDissipationEnergyIncompressible_GradConfig)
{
    // set inputs
    Teuchos::RCP<Teuchos::ParameterList> tInputs =
        Teuchos::getParametersFromXmlString(
            "<ParameterList name='Problem'>"
            "  <ParameterList  name='Criteria'>"
            "    <Parameter  name='Type'    type='string'    value='Scalar Function'/>"
            "    <ParameterList name='My Criteria'>"
            "      <Parameter  name='Flow'                 type='string'    value='Incompressible'/>"
            "      <Parameter  name='Type'                 type='string'    value='Scalar Function'/>"
            "      <Parameter  name='Scalar Function Type' type='string'    value='Internal Dissipation Energy'/>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList  name='Dimensionless Properties'>"
            "    <Parameter  name='Darcy Number'    type='double'    value='1.0'/>"
            "    <Parameter  name='Prandtl Number'  type='double'    value='1.0'/>"
            "  </ParameterList>"
            "</ParameterList>"
            );

    // build mesh, spatial domain, and spatial model
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);
    auto tMeshSets = PlatoUtestHelpers::get_box_mesh_sets(tMesh.operator*());
    Plato::SpatialDomain tDomain(tMesh.operator*(), tMeshSets, "box");
    tDomain.cellOrdinals("body");
    Plato::SpatialModel tModel(tMesh.operator*(), tMeshSets);
    tModel.append(tDomain);

    // set current state
    Plato::Variables tPrimal;
    auto tNumCells = tMesh->nelems();
    auto tNumNodes = tMesh->nverts();
    auto tNumVelDofs = tNumNodes * tMesh->dim();
    Plato::ScalarVector tControl("controls", tNumNodes);
    Plato::blas1::fill(0.5, tControl);
    Plato::ScalarVector tCurVel("current velocity", tNumVelDofs);
    auto tHostCurVel = Kokkos::create_mirror(tCurVel);
    tHostCurVel(0, 0) = 0.12; tHostCurVel(1, 0) = 0.22;
    tHostCurVel(0, 1) = 0.41; tHostCurVel(1, 1) = 0.47;
    tHostCurVel(0, 2) = 0.25; tHostCurVel(1, 2) = 0.86;
    tHostCurVel(0, 3) = 0.15; tHostCurVel(1, 3) = 0.57;
    tHostCurVel(0, 4) = 0.12; tHostCurVel(1, 4) = 0.18;
    tHostCurVel(0, 5) = 0.43; tHostCurVel(1, 5) = 0.11;
    Kokkos::deep_copy(tCurVel, tHostCurVel);
    tPrimal.vector("current velocity", tCurVel);
    Plato::ScalarVector tCurPress("current pressure", tNumNodes);
    Plato::blas1::fill(0.1, tCurPress);
    tPrimal.vector("current pressure", tCurPress);
    Plato::ScalarVector tCurTemp("current temperature", tNumNodes);
    Plato::blas1::fill(1.5, tCurTemp);
    tPrimal.vector("current temperature", tCurTemp);
    Plato::ScalarVector tTimeSteps("time steps", tNumNodes);
    Plato::blas1::fill(0.01, tTimeSteps);
    tPrimal.vector("time steps", tTimeSteps);

    // set physics type
    constexpr Plato::OrdinalType tNumSpaceDim = 2;
    using PhysicsT = Plato::IncompressibleFluids<tNumSpaceDim>;

    // build criterion
    Plato::DataMap tDataMap;
    std::string tFuncName("My Criteria");
    Plato::FluidMechanics::PhysicsScalarFunction<PhysicsT>
        tCriterion(tModel, tDataMap, tInputs.operator*(), tFuncName);
    TEST_EQUALITY("My Criteria", tCriterion.name());

    // test criterion value
    auto tTol = 1e-4;
    auto tGradient = tCriterion.gradientConfig(tControl, tPrimal);
    auto tHostGradient = Kokkos::create_mirror(tGradient);
    Kokkos::deep_copy(tHostGradient, tGradient);

    std::vector<Plato::Scalar> tGold = {0.133106, 0.075276, -0.075276, -0.041220};
    for(Plato::OrdinalType tNode = 0; tNode < tNumNodes; tNode++)
    {
        TEST_FLOATING_EQUALITY(tGold[tNode], tHostGradient(tNode), tTol);
        //printf("Results(Node=%d)=%f\n", tNode, tHostGradient(tNode));
    }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, AverageSurfacePressure_Value)
{
    // set inputs
    Teuchos::RCP<Teuchos::ParameterList> tInputs =
        Teuchos::getParametersFromXmlString(
            "<ParameterList name='Problem'>"
            "  <ParameterList  name='Criteria'>"
            "    <Parameter  name='Type'    type='string'    value='Scalar Function'/>"
            "    <ParameterList name='My Criteria'>"
            "      <Parameter  name='Type'                 type='string'        value='Scalar Function'/>"
            "      <Parameter  name='Sides'                type='Array(string)' value='{x+}'/>"
            "      <Parameter  name='Scalar Function Type' type='string'        value='Average Surface Pressure'/>"
            "    </ParameterList>"
            "  </ParameterList>"
            "</ParameterList>"
            );

    // build mesh, spatial domain, and spatial model
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);
    auto tMeshSets = PlatoUtestHelpers::get_box_mesh_sets(tMesh.operator*());
    Plato::SpatialDomain tDomain(tMesh.operator*(), tMeshSets, "box");
    tDomain.cellOrdinals("body");
    Plato::SpatialModel tModel(tMesh.operator*(), tMeshSets);
    tModel.append(tDomain);

    // set current state
    Plato::Variables tPrimal;
    auto tNumCells = tMesh->nelems();
    auto tNumNodes = tMesh->nverts();
    auto tNumVelDofs = tNumNodes * tMesh->dim();
    Plato::ScalarVector tControl("controls", tNumNodes);
    Plato::blas1::fill(0.5, tControl);
    Plato::ScalarVector tCurVel("current velocity", tNumVelDofs);
    Plato::blas1::fill(1.0, tCurVel);
    tPrimal.vector("current velocity", tCurVel);
    Plato::ScalarVector tCurPress("current pressure", tNumNodes);
    Plato::blas1::fill(0.1, tCurPress);
    tPrimal.vector("current pressure", tCurPress);
    Plato::ScalarVector tCurTemp("current temperature", tNumNodes);
    Plato::blas1::fill(1.5, tCurTemp);
    tPrimal.vector("current temperature", tCurTemp);
    Plato::ScalarVector tTimeSteps("time steps", tNumNodes);
    Plato::blas1::fill(0.01, tTimeSteps);
    tPrimal.vector("time steps", tTimeSteps);

    // set physics type
    constexpr Plato::OrdinalType tNumSpaceDim = 2;
    using PhysicsT = Plato::IncompressibleFluids<tNumSpaceDim>;

    // build criterion
    Plato::DataMap tDataMap;
    std::string tFuncName("My Criteria");
    Plato::FluidMechanics::PhysicsScalarFunction<PhysicsT>
        tCriterion(tModel, tDataMap, tInputs.operator*(), tFuncName);
    TEST_EQUALITY("My Criteria", tCriterion.name());

    // test criterion value
    auto tTol = 1e-4;
    auto tValue = tCriterion.value(tControl, tPrimal);
    TEST_FLOATING_EQUALITY(0.133333, tValue, tTol);
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, AverageSurfacePressure_GradCurPress)
{
    // set inputs
    Teuchos::RCP<Teuchos::ParameterList> tInputs =
        Teuchos::getParametersFromXmlString(
            "<ParameterList name='Problem'>"
            "  <ParameterList  name='Criteria'>"
            "    <Parameter  name='Type'    type='string'    value='Scalar Function'/>"
            "    <ParameterList name='My Criteria'>"
            "      <Parameter  name='Type'                 type='string'        value='Scalar Function'/>"
            "      <Parameter  name='Sides'                type='Array(string)' value='{x+}'/>"
            "      <Parameter  name='Scalar Function Type' type='string'        value='Average Surface Pressure'/>"
            "    </ParameterList>"
            "  </ParameterList>"
            "</ParameterList>"
            );

    // build mesh, spatial domain, and spatial model
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);
    auto tMeshSets = PlatoUtestHelpers::get_box_mesh_sets(tMesh.operator*());
    Plato::SpatialDomain tDomain(tMesh.operator*(), tMeshSets, "box");
    tDomain.cellOrdinals("body");
    Plato::SpatialModel tModel(tMesh.operator*(), tMeshSets);
    tModel.append(tDomain);

    // set current state
    Plato::Variables tPrimal;
    auto tNumCells = tMesh->nelems();
    auto tNumNodes = tMesh->nverts();
    auto tNumVelDofs = tNumNodes * tMesh->dim();
    Plato::ScalarVector tControl("controls", tNumNodes);
    Plato::blas1::fill(0.5, tControl);
    Plato::ScalarVector tCurVel("current velocity", tNumVelDofs);
    Plato::blas1::fill(1.0, tCurVel);
    tPrimal.vector("current velocity", tCurVel);
    Plato::ScalarVector tCurPress("current pressure", tNumNodes);
    Plato::blas1::fill(0.1, tCurPress);
    tPrimal.vector("current pressure", tCurPress);
    Plato::ScalarVector tCurTemp("current temperature", tNumNodes);
    Plato::blas1::fill(1.5, tCurTemp);
    tPrimal.vector("current temperature", tCurTemp);
    Plato::ScalarVector tTimeSteps("time steps", tNumNodes);
    Plato::blas1::fill(0.01, tTimeSteps);
    tPrimal.vector("time steps", tTimeSteps);

    // set physics type
    constexpr Plato::OrdinalType tNumSpaceDim = 2;
    using PhysicsT = Plato::IncompressibleFluids<tNumSpaceDim>;

    // build criterion
    Plato::DataMap tDataMap;
    std::string tFuncName("My Criteria");
    Plato::FluidMechanics::PhysicsScalarFunction<PhysicsT>
        tCriterion(tModel, tDataMap, tInputs.operator*(), tFuncName);
    TEST_EQUALITY("My Criteria", tCriterion.name());

    // test criterion value
    auto tGradCurPress = tCriterion.gradientCurrentPress(tControl, tPrimal);
    auto tHostGradCurPress = Kokkos::create_mirror(tGradCurPress);
    Kokkos::deep_copy(tHostGradCurPress, tGradCurPress);

    auto tTol = 1e-6;
    std::vector<Plato::Scalar> tGold = {0.5, 0.0, 0.333333333, 0.5};
    for(Plato::OrdinalType tNode = 0; tNode < tNumNodes; tNode++)
    {
        TEST_FLOATING_EQUALITY(tGold[tNode], tHostGradCurPress(tNode), tTol);
    }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, AverageSurfaceTemperature_Value)
{
    // set inputs
    Teuchos::RCP<Teuchos::ParameterList> tInputs =
        Teuchos::getParametersFromXmlString(
            "<ParameterList name='Problem'>"
            "  <ParameterList  name='Criteria'>"
            "    <Parameter  name='Type'    type='string'    value='Scalar Function'/>"
            "    <ParameterList name='My Criteria'>"
            "      <Parameter  name='Type'                 type='string'        value='Scalar Function'/>"
            "      <Parameter  name='Sides'                type='Array(string)' value='{x+}'/>"
            "      <Parameter  name='Scalar Function Type' type='string'        value='Average Surface Temperature'/>"
            "    </ParameterList>"
            "  </ParameterList>"
            "</ParameterList>"
            );

    // build mesh, spatial domain, and spatial model
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);
    auto tMeshSets = PlatoUtestHelpers::get_box_mesh_sets(tMesh.operator*());
    Plato::SpatialDomain tDomain(tMesh.operator*(), tMeshSets, "box");
    tDomain.cellOrdinals("body");
    Plato::SpatialModel tModel(tMesh.operator*(), tMeshSets);
    tModel.append(tDomain);

    // set current state
    Plato::Variables tPrimal;
    auto tNumCells = tMesh->nelems();
    auto tNumNodes = tMesh->nverts();
    auto tNumVelDofs = tNumNodes * tMesh->dim();
    Plato::ScalarVector tControl("controls", tNumNodes);
    Plato::blas1::fill(0.5, tControl);
    Plato::ScalarVector tCurVel("current velocity", tNumVelDofs);
    Plato::blas1::fill(1.0, tCurVel);
    tPrimal.vector("current velocity", tCurVel);
    Plato::ScalarVector tCurPress("current pressure", tNumNodes);
    Plato::blas1::fill(0.1, tCurPress);
    tPrimal.vector("current pressure", tCurPress);
    Plato::ScalarVector tCurTemp("current temperature", tNumNodes);
    Plato::blas1::fill(1.5, tCurTemp);
    tPrimal.vector("current temperature", tCurTemp);
    Plato::ScalarVector tTimeSteps("time steps", tNumNodes);
    Plato::blas1::fill(0.01, tTimeSteps);
    tPrimal.vector("time steps", tTimeSteps);

    // set physics type
    constexpr Plato::OrdinalType tNumSpaceDim = 2;
    using PhysicsT = Plato::IncompressibleFluids<tNumSpaceDim>;

    // build criterion
    Plato::DataMap tDataMap;
    std::string tFuncName("My Criteria");
    Plato::FluidMechanics::PhysicsScalarFunction<PhysicsT>
        tCriterion(tModel, tDataMap, tInputs.operator*(), tFuncName);
    TEST_EQUALITY("My Criteria", tCriterion.name());

    // test criterion value
    auto tTol = 1e-6;
    auto tValue = tCriterion.value(tControl, tPrimal);
    TEST_FLOATING_EQUALITY(2.0, tValue, tTol);
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, AverageSurfaceTemperature_GradCurTemp)
{
    // set inputs
    Teuchos::RCP<Teuchos::ParameterList> tInputs =
        Teuchos::getParametersFromXmlString(
            "<ParameterList name='Problem'>"
            "  <ParameterList  name='Criteria'>"
            "    <Parameter  name='Type'    type='string'    value='Scalar Function'/>"
            "    <ParameterList name='My Criteria'>"
            "      <Parameter  name='Type'                 type='string'        value='Scalar Function'/>"
            "      <Parameter  name='Sides'                type='Array(string)' value='{x+}'/>"
            "      <Parameter  name='Scalar Function Type' type='string'        value='Average Surface Temperature'/>"
            "    </ParameterList>"
            "  </ParameterList>"
            "</ParameterList>"
            );

    // build mesh, spatial domain, and spatial model
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);
    auto tMeshSets = PlatoUtestHelpers::get_box_mesh_sets(tMesh.operator*());
    Plato::SpatialDomain tDomain(tMesh.operator*(), tMeshSets, "box");
    tDomain.cellOrdinals("body");
    Plato::SpatialModel tModel(tMesh.operator*(), tMeshSets);
    tModel.append(tDomain);

    // set current state
    Plato::Variables tPrimal;
    auto tNumCells = tMesh->nelems();
    auto tNumNodes = tMesh->nverts();
    auto tNumVelDofs = tNumNodes * tMesh->dim();
    Plato::ScalarVector tControl("controls", tNumNodes);
    Plato::blas1::fill(0.5, tControl);
    Plato::ScalarVector tCurVel("current velocity", tNumVelDofs);
    Plato::blas1::fill(1.0, tCurVel);
    tPrimal.vector("current velocity", tCurVel);
    Plato::ScalarVector tCurPress("current pressure", tNumNodes);
    Plato::blas1::fill(0.1, tCurPress);
    tPrimal.vector("current pressure", tCurPress);
    Plato::ScalarVector tCurTemp("current temperature", tNumNodes);
    Plato::blas1::fill(1.5, tCurTemp);
    tPrimal.vector("current temperature", tCurTemp);
    Plato::ScalarVector tTimeSteps("time steps", tNumNodes);
    Plato::blas1::fill(0.01, tTimeSteps);
    tPrimal.vector("time steps", tTimeSteps);

    // set physics type
    constexpr Plato::OrdinalType tNumSpaceDim = 2;
    using PhysicsT = Plato::IncompressibleFluids<tNumSpaceDim>;

    // build criterion
    Plato::DataMap tDataMap;
    std::string tFuncName("My Criteria");
    Plato::FluidMechanics::PhysicsScalarFunction<PhysicsT>
        tCriterion(tModel, tDataMap, tInputs.operator*(), tFuncName);
    TEST_EQUALITY("My Criteria", tCriterion.name());

    // test criterion value
    auto tGradCurTemp = tCriterion.gradientCurrentTemp(tControl, tPrimal);
    auto tHostGradCurTemp = Kokkos::create_mirror(tGradCurTemp);
    Kokkos::deep_copy(tHostGradCurTemp, tGradCurTemp);

    auto tTol = 1e-6;
    std::vector<Plato::Scalar> tGold = {0.5, 0.0, 0.333333333, 0.5};
    for(Plato::OrdinalType tNode = 0; tNode < tNumNodes; tNode++)
    {
        TEST_FLOATING_EQUALITY(tGold[tNode], tHostGradCurTemp(tNode), tTol);
    }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, BuildScalarFunctionWorksets_SpatialDomain)
{
    // build mesh and spatial domain
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);
    auto tMeshSets = PlatoUtestHelpers::get_box_mesh_sets(tMesh.operator*());
    Plato::SpatialDomain tDomain(tMesh.operator*(), tMeshSets, "box");
    tDomain.cellOrdinals("body");

    // set physics and evaluation type
    constexpr Plato::OrdinalType tNumSpaceDim = 2;
    using PhysicsT = Plato::IncompressibleFluids<tNumSpaceDim>;
    using ResidualEvalT = Plato::FluidMechanics::Evaluation<PhysicsT::SimplexT>::Residual;

    // set current state
    Plato::Variables tPrimal;
    auto tNumCells = tMesh->nelems();
    auto tNumNodes = tMesh->nverts();
    auto tNumVelDofs = tNumNodes * tNumSpaceDim;
    Plato::ScalarVector tControls("controls", tNumNodes);
    Plato::blas1::fill(0.5, tControls);
    Plato::ScalarVector tCurVel("current velocity", tNumVelDofs);
    Plato::blas1::fill(1.0, tCurVel);
    tPrimal.vector("current velocity", tCurVel);
    Plato::ScalarVector tCurPress("current pressure", tNumNodes);
    Plato::blas1::fill(2.0, tCurPress);
    tPrimal.vector("current pressure", tCurPress);
    Plato::ScalarVector tCurTemp("current temperature", tNumNodes);
    Plato::blas1::fill(3.0, tCurTemp);
    tPrimal.vector("current temperature", tCurTemp);
    Plato::ScalarVector tTimeSteps("time steps", tNumNodes);
    Plato::blas1::fill(4.0, tTimeSteps);
    tPrimal.vector("time steps", tTimeSteps);

    // call build_scalar_function_worksets
    Plato::WorkSets tWorkSets;
    Plato::LocalOrdinalMaps<PhysicsT> tOrdinalMaps(*tMesh);
    Plato::FluidMechanics::build_scalar_function_worksets<ResidualEvalT>
        (tDomain, tControls, tPrimal, tOrdinalMaps, tWorkSets);

    // test current velocity results
    auto tCurVelWS = Plato::metadata<Plato::ScalarMultiVectorT<ResidualEvalT::CurrentMomentumScalarType>>(tWorkSets.get("current velocity"));
    TEST_EQUALITY(tNumCells, tCurVelWS.extent(0));
    auto tNumVelDofsPerCell = PhysicsT::mNumMomentumDofsPerCell;
    TEST_EQUALITY(tNumVelDofsPerCell, tCurVelWS.extent(1));
    auto tHostCurVelWS = Kokkos::create_mirror(tCurVelWS);
    Kokkos::deep_copy(tHostCurVelWS, tCurVelWS);
    const Plato::Scalar tTol = 1e-6;
    for (decltype(tNumCells) tCell = 0; tCell < tNumCells; tCell++)
    {
        for (decltype(tNumVelDofsPerCell) tDof = 0; tDof < tNumVelDofsPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(1.0, tHostCurVelWS(tCell, tDof), tTol);
        }
    }

    // test current pressure results
    auto tCurPressWS = Plato::metadata<Plato::ScalarMultiVectorT<ResidualEvalT::CurrentMassScalarType>>(tWorkSets.get("current pressure"));
    TEST_EQUALITY(tNumCells, tCurPressWS.extent(0));
    auto tNumPressDofsPerCell = PhysicsT::mNumMassDofsPerCell;
    TEST_EQUALITY(tNumPressDofsPerCell, tCurPressWS.extent(1));
    auto tHostCurPressWS = Kokkos::create_mirror(tCurPressWS);
    Kokkos::deep_copy(tHostCurPressWS, tCurPressWS);
    for (decltype(tNumCells) tCell = 0; tCell < tNumCells; tCell++)
    {
        for (decltype(tNumPressDofsPerCell) tDof = 0; tDof < tNumPressDofsPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(2.0, tHostCurPressWS(tCell, tDof), tTol);
        }
    }

    // test current temperature results
    auto tCurTempWS = Plato::metadata<Plato::ScalarMultiVectorT<ResidualEvalT::CurrentEnergyScalarType>>(tWorkSets.get("current temperature"));
    TEST_EQUALITY(tNumCells, tCurTempWS.extent(0));
    auto tNumTempDofsPerCell = PhysicsT::mNumEnergyDofsPerCell;
    TEST_EQUALITY(tNumTempDofsPerCell, tCurTempWS.extent(1));
    auto tHostCurTempWS = Kokkos::create_mirror(tCurTempWS);
    Kokkos::deep_copy(tHostCurTempWS, tCurTempWS);
    for (decltype(tNumCells) tCell = 0; tCell < tNumCells; tCell++)
    {
        for (decltype(tNumTempDofsPerCell) tDof = 0; tDof < tNumTempDofsPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(3.0, tHostCurTempWS(tCell, tDof), tTol);
        }
    }

    // test time steps results
    auto tTimeStepWS = Plato::metadata<Plato::ScalarMultiVector>(tWorkSets.get("time steps"));
    TEST_EQUALITY(tNumCells, tTimeStepWS.extent(0));
    auto tNumNodesPerCell = PhysicsT::mNumNodesPerCell;
    TEST_EQUALITY(tNumNodesPerCell, tTimeStepWS.extent(1));
    auto tHostTimeStepWS = Kokkos::create_mirror(tTimeStepWS);
    Kokkos::deep_copy(tHostTimeStepWS, tTimeStepWS);
    for (decltype(tNumCells) tCell = 0; tCell < tNumCells; tCell++)
    {
        for (decltype(tNumNodesPerCell) tDof = 0; tDof < tNumNodesPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(4.0, tHostTimeStepWS(tCell, tDof), tTol);
        }
    }

    // test controls results
    auto tControlWS = Plato::metadata<Plato::ScalarMultiVectorT<ResidualEvalT::ControlScalarType>>(tWorkSets.get("control"));
    TEST_EQUALITY(tNumCells, tControlWS.extent(0));
    TEST_EQUALITY(tNumNodesPerCell, tControlWS.extent(1));
    auto tHostControlWS = Kokkos::create_mirror(tControlWS);
    Kokkos::deep_copy(tHostControlWS, tControlWS);
    for (decltype(tNumCells) tCell = 0; tCell < tNumCells; tCell++)
    {
        for (decltype(tNumNodesPerCell) tDof = 0; tDof < tNumNodesPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(0.5, tHostControlWS(tCell, tDof), tTol);
        }
    }

    // test configuration results
    auto tConfigWS = Plato::metadata<Plato::ScalarArray3DT<ResidualEvalT::ConfigScalarType>>(tWorkSets.get("configuration"));
    TEST_EQUALITY(tNumCells, tConfigWS.extent(0));
    TEST_EQUALITY(tNumNodesPerCell, tConfigWS.extent(1));
    auto tNumConfigDofsPerNode = PhysicsT::mNumConfigDofsPerNode;
    TEST_EQUALITY(tNumConfigDofsPerNode, tConfigWS.extent(2));
    auto tHostConfigWS = Kokkos::create_mirror(tConfigWS);
    Kokkos::deep_copy(tHostConfigWS, tConfigWS);
    Plato::ScalarArray3D tGoldConfigWS("gold configuration", tNumCells, tNumNodesPerCell, tNumConfigDofsPerNode);
    auto tHostGoldConfigWS = Kokkos::create_mirror(tGoldConfigWS);
    tHostGoldConfigWS(0,0,0) = 0; tHostGoldConfigWS(0,1,0) = 1; tHostGoldConfigWS(0,2,0) = 1;
    tHostGoldConfigWS(1,0,0) = 1; tHostGoldConfigWS(1,1,0) = 0; tHostGoldConfigWS(1,2,0) = 0;
    tHostGoldConfigWS(0,0,1) = 0; tHostGoldConfigWS(0,1,1) = 0; tHostGoldConfigWS(0,2,1) = 1;
    tHostGoldConfigWS(1,0,1) = 1; tHostGoldConfigWS(1,1,1) = 1; tHostGoldConfigWS(1,2,1) = 0;
    for (decltype(tNumCells) tCell = 0; tCell < tNumCells; tCell++)
    {
        for (decltype(tNumNodesPerCell) tNode = 0; tNode < tNumNodesPerCell; tNode++)
        {
            for (decltype(tNumConfigDofsPerNode) tDof = 0; tDof < tNumConfigDofsPerNode; tDof++)
            {
                TEST_FLOATING_EQUALITY(tHostGoldConfigWS(tCell, tNode, tDof), tHostConfigWS(tCell, tNode, tDof), tTol);
            }
        }
    }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, BuildVectorFunctionWorksets_SpatialDomain)
{
    // build mesh and spatial domain
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);
    auto tMeshSets = PlatoUtestHelpers::get_box_mesh_sets(tMesh.operator*());
    Plato::SpatialDomain tDomain(tMesh.operator*(), tMeshSets, "box");
    tDomain.cellOrdinals("body");

    // set physics and evaluation type
    constexpr Plato::OrdinalType tNumSpaceDim = 2;
    using PhysicsT = Plato::IncompressibleFluids<tNumSpaceDim>;
    using ResidualEvalT = Plato::FluidMechanics::Evaluation<PhysicsT::SimplexT>::Residual;

    // set current state
    Plato::Variables tPrimal;
    auto tNumCells = tMesh->nelems();
    auto tNumNodes = tMesh->nverts();
    auto tNumVelDofs = tNumNodes * tNumSpaceDim;
    Plato::ScalarVector tControls("controls", tNumNodes);
    Plato::blas1::fill(0.5, tControls);
    Plato::ScalarVector tCurPred("current predictor", tNumVelDofs);
    Plato::blas1::fill(0.1, tCurPred);
    tPrimal.vector("current predictor", tCurPred);
    Plato::ScalarVector tCurVel("current velocity", tNumVelDofs);
    Plato::blas1::fill(1.0, tCurVel);
    tPrimal.vector("current velocity", tCurVel);
    Plato::ScalarVector tCurPress("current pressure", tNumNodes);
    Plato::blas1::fill(2.0, tCurPress);
    tPrimal.vector("current pressure", tCurPress);
    Plato::ScalarVector tCurTemp("current temperature", tNumNodes);
    Plato::blas1::fill(3.0, tCurTemp);
    tPrimal.vector("current temperature", tCurTemp);
    Plato::ScalarVector tPrevVel("previous velocity", tNumVelDofs);
    Plato::blas1::fill(0.8, tPrevVel);
    tPrimal.vector("previous velocity", tPrevVel);
    Plato::ScalarVector tPrevPress("previous pressure", tNumNodes);
    Plato::blas1::fill(1.8, tPrevPress);
    tPrimal.vector("previous pressure", tPrevPress);
    Plato::ScalarVector tPrevTemp("previous temperature", tNumNodes);
    Plato::blas1::fill(2.8, tPrevTemp);
    tPrimal.vector("previous temperature", tPrevTemp);
    Plato::ScalarVector tTimeSteps("time steps", tNumNodes);
    Plato::blas1::fill(4.0, tTimeSteps);
    tPrimal.vector("time steps", tTimeSteps);
    Plato::ScalarVector tArtCompress("artificial compressibility", tNumNodes);
    Plato::blas1::fill(5.0, tArtCompress);
    tPrimal.vector("artificial compressibility", tArtCompress);
    Plato::ScalarVector tVelBCs("prescribed velocity", tNumVelDofs);
    Plato::blas1::fill(6.0, tVelBCs);
    tPrimal.vector("prescribed velocity", tVelBCs);

    // call build_vector_function_worksets
    Plato::WorkSets tWorkSets;
    Plato::LocalOrdinalMaps<PhysicsT> tOrdinalMaps(*tMesh);
    Plato::FluidMechanics::build_vector_function_worksets<ResidualEvalT>
        (tNumCells, tControls, tPrimal, tOrdinalMaps, tWorkSets);

    // test current velocity results
    auto tCurVelWS = Plato::metadata<Plato::ScalarMultiVectorT<ResidualEvalT::CurrentMomentumScalarType>>(tWorkSets.get("current velocity"));
    TEST_EQUALITY(tNumCells, tCurVelWS.extent(0));
    auto tNumVelDofsPerCell = PhysicsT::mNumMomentumDofsPerCell;
    TEST_EQUALITY(tNumVelDofsPerCell, tCurVelWS.extent(1));
    auto tHostCurVelWS = Kokkos::create_mirror(tCurVelWS);
    Kokkos::deep_copy(tHostCurVelWS, tCurVelWS);
    const Plato::Scalar tTol = 1e-6;
    for (decltype(tNumCells) tCell = 0; tCell < tNumCells; tCell++)
    {
        for (decltype(tNumVelDofsPerCell) tDof = 0; tDof < tNumVelDofsPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(1.0, tHostCurVelWS(tCell, tDof), tTol);
        }
    }

    // test current pressure results
    auto tCurPressWS = Plato::metadata<Plato::ScalarMultiVectorT<ResidualEvalT::CurrentMassScalarType>>(tWorkSets.get("current pressure"));
    TEST_EQUALITY(tNumCells, tCurPressWS.extent(0));
    auto tNumPressDofsPerCell = PhysicsT::mNumMassDofsPerCell;
    TEST_EQUALITY(tNumPressDofsPerCell, tCurPressWS.extent(1));
    auto tHostCurPressWS = Kokkos::create_mirror(tCurPressWS);
    Kokkos::deep_copy(tHostCurPressWS, tCurPressWS);
    for (decltype(tNumCells) tCell = 0; tCell < tNumCells; tCell++)
    {
        for (decltype(tNumPressDofsPerCell) tDof = 0; tDof < tNumPressDofsPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(2.0, tHostCurPressWS(tCell, tDof), tTol);
        }
    }

    // test current temperature results
    auto tCurTempWS = Plato::metadata<Plato::ScalarMultiVectorT<ResidualEvalT::CurrentEnergyScalarType>>(tWorkSets.get("current temperature"));
    TEST_EQUALITY(tNumCells, tCurTempWS.extent(0));
    auto tNumTempDofsPerCell = PhysicsT::mNumEnergyDofsPerCell;
    TEST_EQUALITY(tNumTempDofsPerCell, tCurTempWS.extent(1));
    auto tHostCurTempWS = Kokkos::create_mirror(tCurTempWS);
    Kokkos::deep_copy(tHostCurTempWS, tCurTempWS);
    for (decltype(tNumCells) tCell = 0; tCell < tNumCells; tCell++)
    {
        for (decltype(tNumTempDofsPerCell) tDof = 0; tDof < tNumTempDofsPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(3.0, tHostCurTempWS(tCell, tDof), tTol);
        }
    }

    // test previous velocity results
    auto tPrevVelWS = Plato::metadata<Plato::ScalarMultiVectorT<ResidualEvalT::PreviousMomentumScalarType>>(tWorkSets.get("previous velocity"));
    TEST_EQUALITY(tNumCells, tPrevVelWS.extent(0));
    TEST_EQUALITY(tNumVelDofsPerCell, tPrevVelWS.extent(1));
    auto tHostPrevVelWS = Kokkos::create_mirror(tPrevVelWS);
    Kokkos::deep_copy(tHostPrevVelWS, tPrevVelWS);
    for (decltype(tNumCells) tCell = 0; tCell < tNumCells; tCell++)
    {
        for (decltype(tNumVelDofsPerCell) tDof = 0; tDof < tNumVelDofsPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(0.8, tHostPrevVelWS(tCell, tDof), tTol);
        }
    }

    // test previous pressure results
    auto tPrevPressWS = Plato::metadata<Plato::ScalarMultiVectorT<ResidualEvalT::PreviousMassScalarType>>(tWorkSets.get("previous pressure"));
    TEST_EQUALITY(tNumCells, tPrevPressWS.extent(0));
    TEST_EQUALITY(tNumPressDofsPerCell, tPrevPressWS.extent(1));
    auto tHostPrevPressWS = Kokkos::create_mirror(tPrevPressWS);
    Kokkos::deep_copy(tHostPrevPressWS, tPrevPressWS);
    for (decltype(tNumCells) tCell = 0; tCell < tNumCells; tCell++)
    {
        for (decltype(tNumPressDofsPerCell) tDof = 0; tDof < tNumPressDofsPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(1.8, tHostPrevPressWS(tCell, tDof), tTol);
        }
    }

    // test previous temperature results
    auto tPrevTempWS = Plato::metadata<Plato::ScalarMultiVectorT<ResidualEvalT::PreviousEnergyScalarType>>(tWorkSets.get("previous temperature"));
    TEST_EQUALITY(tNumCells, tPrevTempWS.extent(0));
    TEST_EQUALITY(tNumTempDofsPerCell, tPrevTempWS.extent(1));
    auto tHostPrevTempWS = Kokkos::create_mirror(tPrevTempWS);
    Kokkos::deep_copy(tHostPrevTempWS, tPrevTempWS);
    for (decltype(tNumCells) tCell = 0; tCell < tNumCells; tCell++)
    {
        for (decltype(tNumTempDofsPerCell) tDof = 0; tDof < tNumTempDofsPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(2.8, tHostPrevTempWS(tCell, tDof), tTol);
        }
    }

    // test time steps results
    auto tTimeStepWS = Plato::metadata<Plato::ScalarMultiVector>(tWorkSets.get("time steps"));
    TEST_EQUALITY(tNumCells, tTimeStepWS.extent(0));
    auto tNumNodesPerCell = PhysicsT::mNumNodesPerCell;
    TEST_EQUALITY(tNumNodesPerCell, tTimeStepWS.extent(1));
    auto tHostTimeStepWS = Kokkos::create_mirror(tTimeStepWS);
    Kokkos::deep_copy(tHostTimeStepWS, tTimeStepWS);
    for (decltype(tNumCells) tCell = 0; tCell < tNumCells; tCell++)
    {
        for (decltype(tNumNodesPerCell) tDof = 0; tDof < tNumNodesPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(4.0, tHostTimeStepWS(tCell, tDof), tTol);
        }
    }

    // test artificial compressibility results
    auto tArtCompressWS = Plato::metadata<Plato::ScalarMultiVector>(tWorkSets.get("artificial compressibility"));
    TEST_EQUALITY(tNumCells, tArtCompressWS.extent(0));
    TEST_EQUALITY(tNumNodesPerCell, tArtCompressWS.extent(1));
    auto tHostArtCompressWS = Kokkos::create_mirror(tArtCompressWS);
    Kokkos::deep_copy(tHostArtCompressWS, tArtCompressWS);
    for (decltype(tNumCells) tCell = 0; tCell < tNumCells; tCell++)
    {
        for (decltype(tNumNodesPerCell) tDof = 0; tDof < tNumNodesPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(5.0, tHostArtCompressWS(tCell, tDof), tTol);
        }
    }

    // test prescribed velocity results
    auto tVelBCsWS = Plato::metadata<Plato::ScalarMultiVector>(tWorkSets.get("prescribed velocity"));
    TEST_EQUALITY(tNumCells, tVelBCsWS.extent(0));
    TEST_EQUALITY(tNumVelDofsPerCell, tVelBCsWS.extent(1));
    auto tHostVelBCsWS = Kokkos::create_mirror(tVelBCsWS);
    Kokkos::deep_copy(tHostVelBCsWS, tVelBCsWS);
    for (decltype(tNumCells) tCell = 0; tCell < tNumCells; tCell++)
    {
        for (decltype(tNumVelDofsPerCell) tDof = 0; tDof < tNumVelDofsPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(6.0, tHostVelBCsWS(tCell, tDof), tTol);
        }
    }

    // test controls results
    auto tControlWS = Plato::metadata<Plato::ScalarMultiVectorT<ResidualEvalT::ControlScalarType>>(tWorkSets.get("control"));
    TEST_EQUALITY(tNumCells, tControlWS.extent(0));
    TEST_EQUALITY(tNumNodesPerCell, tControlWS.extent(1));
    auto tHostControlWS = Kokkos::create_mirror(tControlWS);
    Kokkos::deep_copy(tHostControlWS, tControlWS);
    for (decltype(tNumCells) tCell = 0; tCell < tNumCells; tCell++)
    {
        for (decltype(tNumNodesPerCell) tDof = 0; tDof < tNumNodesPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(0.5, tHostControlWS(tCell, tDof), tTol);
        }
    }

    // test configuration results
    auto tConfigWS = Plato::metadata<Plato::ScalarArray3DT<ResidualEvalT::ConfigScalarType>>(tWorkSets.get("configuration"));
    TEST_EQUALITY(tNumCells, tConfigWS.extent(0));
    TEST_EQUALITY(tNumNodesPerCell, tConfigWS.extent(1));
    auto tNumConfigDofsPerNode = PhysicsT::mNumConfigDofsPerNode;
    TEST_EQUALITY(tNumConfigDofsPerNode, tConfigWS.extent(2));
    auto tHostConfigWS = Kokkos::create_mirror(tConfigWS);
    Kokkos::deep_copy(tHostConfigWS, tConfigWS);
    Plato::ScalarArray3D tGoldConfigWS("gold configuration", tNumCells, tNumNodesPerCell, tNumConfigDofsPerNode);
    auto tHostGoldConfigWS = Kokkos::create_mirror(tGoldConfigWS);
    tHostGoldConfigWS(0,0,0) = 0; tHostGoldConfigWS(0,1,0) = 1; tHostGoldConfigWS(0,2,0) = 1;
    tHostGoldConfigWS(1,0,0) = 1; tHostGoldConfigWS(1,1,0) = 0; tHostGoldConfigWS(1,2,0) = 0;
    tHostGoldConfigWS(0,0,1) = 0; tHostGoldConfigWS(0,1,1) = 0; tHostGoldConfigWS(0,2,1) = 1;
    tHostGoldConfigWS(1,0,1) = 1; tHostGoldConfigWS(1,1,1) = 1; tHostGoldConfigWS(1,2,1) = 0;
    for (decltype(tNumCells) tCell = 0; tCell < tNumCells; tCell++)
    {
        for (decltype(tNumNodesPerCell) tNode = 0; tNode < tNumNodesPerCell; tNode++)
        {
            for (decltype(tNumConfigDofsPerNode) tDof = 0; tDof < tNumConfigDofsPerNode; tDof++)
            {
                TEST_FLOATING_EQUALITY(tHostGoldConfigWS(tCell, tNode, tDof), tHostConfigWS(tCell, tNode, tDof), tTol);
            }
        }
    }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, BuildVectorFunctionWorksets)
{
    constexpr Plato::OrdinalType tNumSpaceDim = 2;
    using PhysicsT = Plato::IncompressibleFluids<tNumSpaceDim>;
    using ResidualEvalT = Plato::FluidMechanics::Evaluation<PhysicsT::SimplexT>::Residual;

    // set current state
    Plato::Variables tPrimal;
    auto tNumCells = 2;
    auto tNumNodes = 4;
    auto tNumVelDofs = tNumNodes * tNumSpaceDim;
    Plato::ScalarVector tControls("controls", tNumNodes);
    Plato::blas1::fill(0.5, tControls);
    Plato::ScalarVector tCurPred("current predictor", tNumVelDofs);
    Plato::blas1::fill(0.1, tCurPred);
    tPrimal.vector("current predictor", tCurPred);
    Plato::ScalarVector tCurVel("current velocity", tNumVelDofs);
    Plato::blas1::fill(1.0, tCurVel);
    tPrimal.vector("current velocity", tCurVel);
    Plato::ScalarVector tCurPress("current pressure", tNumNodes);
    Plato::blas1::fill(2.0, tCurPress);
    tPrimal.vector("current pressure", tCurPress);
    Plato::ScalarVector tCurTemp("current temperature", tNumNodes);
    Plato::blas1::fill(3.0, tCurTemp);
    tPrimal.vector("current temperature", tCurTemp);
    Plato::ScalarVector tPrevVel("previous velocity", tNumVelDofs);
    Plato::blas1::fill(0.8, tPrevVel);
    tPrimal.vector("previous velocity", tPrevVel);
    Plato::ScalarVector tPrevPress("previous pressure", tNumNodes);
    Plato::blas1::fill(1.8, tPrevPress);
    tPrimal.vector("previous pressure", tPrevPress);
    Plato::ScalarVector tPrevTemp("previous temperature", tNumNodes);
    Plato::blas1::fill(2.8, tPrevTemp);
    tPrimal.vector("previous temperature", tPrevTemp);
    Plato::ScalarVector tTimeSteps("time steps", tNumNodes);
    Plato::blas1::fill(4.0, tTimeSteps);
    tPrimal.vector("time steps", tTimeSteps);
    Plato::ScalarVector tArtCompress("artificial compressibility", tNumNodes);
    Plato::blas1::fill(5.0, tArtCompress);
    tPrimal.vector("artificial compressibility", tArtCompress);
    Plato::ScalarVector tVelBCs("prescribed velocity", tNumVelDofs);
    Plato::blas1::fill(6.0, tVelBCs);
    tPrimal.vector("prescribed velocity", tVelBCs);

    // set ordinal maps;
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);
    Plato::LocalOrdinalMaps<PhysicsT> tOrdinalMaps(*tMesh);

    // call build_vector_function_worksets
    Plato::WorkSets tWorkSets;
    Plato::FluidMechanics::build_vector_function_worksets<ResidualEvalT>
        (tNumCells, tControls, tPrimal, tOrdinalMaps, tWorkSets);

    // test current velocity results
    auto tCurVelWS = Plato::metadata<Plato::ScalarMultiVectorT<ResidualEvalT::CurrentMomentumScalarType>>(tWorkSets.get("current velocity"));
    TEST_EQUALITY(tNumCells, tCurVelWS.extent(0));
    auto tNumVelDofsPerCell = PhysicsT::mNumMomentumDofsPerCell;
    TEST_EQUALITY(tNumVelDofsPerCell, tCurVelWS.extent(1));
    auto tHostCurVelWS = Kokkos::create_mirror(tCurVelWS);
    Kokkos::deep_copy(tHostCurVelWS, tCurVelWS);
    const Plato::Scalar tTol = 1e-6;
    for (decltype(tNumCells) tCell = 0; tCell < tNumCells; tCell++)
    {
        for (decltype(tNumVelDofsPerCell) tDof = 0; tDof < tNumVelDofsPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(1.0, tHostCurVelWS(tCell, tDof), tTol);
        }
    }

    // test current pressure results
    auto tCurPressWS = Plato::metadata<Plato::ScalarMultiVectorT<ResidualEvalT::CurrentMassScalarType>>(tWorkSets.get("current pressure"));
    TEST_EQUALITY(tNumCells, tCurPressWS.extent(0));
    auto tNumPressDofsPerCell = PhysicsT::mNumMassDofsPerCell;
    TEST_EQUALITY(tNumPressDofsPerCell, tCurPressWS.extent(1));
    auto tHostCurPressWS = Kokkos::create_mirror(tCurPressWS);
    Kokkos::deep_copy(tHostCurPressWS, tCurPressWS);
    for (decltype(tNumCells) tCell = 0; tCell < tNumCells; tCell++)
    {
        for (decltype(tNumPressDofsPerCell) tDof = 0; tDof < tNumPressDofsPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(2.0, tHostCurPressWS(tCell, tDof), tTol);
        }
    }

    // test current temperature results
    auto tCurTempWS = Plato::metadata<Plato::ScalarMultiVectorT<ResidualEvalT::CurrentEnergyScalarType>>(tWorkSets.get("current temperature"));
    TEST_EQUALITY(tNumCells, tCurTempWS.extent(0));
    auto tNumTempDofsPerCell = PhysicsT::mNumEnergyDofsPerCell;
    TEST_EQUALITY(tNumTempDofsPerCell, tCurTempWS.extent(1));
    auto tHostCurTempWS = Kokkos::create_mirror(tCurTempWS);
    Kokkos::deep_copy(tHostCurTempWS, tCurTempWS);
    for (decltype(tNumCells) tCell = 0; tCell < tNumCells; tCell++)
    {
        for (decltype(tNumTempDofsPerCell) tDof = 0; tDof < tNumTempDofsPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(3.0, tHostCurTempWS(tCell, tDof), tTol);
        }
    }

    // test previous velocity results
    auto tPrevVelWS = Plato::metadata<Plato::ScalarMultiVectorT<ResidualEvalT::PreviousMomentumScalarType>>(tWorkSets.get("previous velocity"));
    TEST_EQUALITY(tNumCells, tPrevVelWS.extent(0));
    TEST_EQUALITY(tNumVelDofsPerCell, tPrevVelWS.extent(1));
    auto tHostPrevVelWS = Kokkos::create_mirror(tPrevVelWS);
    Kokkos::deep_copy(tHostPrevVelWS, tPrevVelWS);
    for (decltype(tNumCells) tCell = 0; tCell < tNumCells; tCell++)
    {
        for (decltype(tNumVelDofsPerCell) tDof = 0; tDof < tNumVelDofsPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(0.8, tHostPrevVelWS(tCell, tDof), tTol);
        }
    }

    // test previous pressure results
    auto tPrevPressWS = Plato::metadata<Plato::ScalarMultiVectorT<ResidualEvalT::PreviousMassScalarType>>(tWorkSets.get("previous pressure"));
    TEST_EQUALITY(tNumCells, tPrevPressWS.extent(0));
    TEST_EQUALITY(tNumPressDofsPerCell, tPrevPressWS.extent(1));
    auto tHostPrevPressWS = Kokkos::create_mirror(tPrevPressWS);
    Kokkos::deep_copy(tHostPrevPressWS, tPrevPressWS);
    for (decltype(tNumCells) tCell = 0; tCell < tNumCells; tCell++)
    {
        for (decltype(tNumPressDofsPerCell) tDof = 0; tDof < tNumPressDofsPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(1.8, tHostPrevPressWS(tCell, tDof), tTol);
        }
    }

    // test previous temperature results
    auto tPrevTempWS = Plato::metadata<Plato::ScalarMultiVectorT<ResidualEvalT::PreviousEnergyScalarType>>(tWorkSets.get("previous temperature"));
    TEST_EQUALITY(tNumCells, tPrevTempWS.extent(0));
    TEST_EQUALITY(tNumTempDofsPerCell, tPrevTempWS.extent(1));
    auto tHostPrevTempWS = Kokkos::create_mirror(tPrevTempWS);
    Kokkos::deep_copy(tHostPrevTempWS, tPrevTempWS);
    for (decltype(tNumCells) tCell = 0; tCell < tNumCells; tCell++)
    {
        for (decltype(tNumTempDofsPerCell) tDof = 0; tDof < tNumTempDofsPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(2.8, tHostPrevTempWS(tCell, tDof), tTol);
        }
    }

    // test time steps results
    auto tTimeStepWS = Plato::metadata<Plato::ScalarMultiVector>(tWorkSets.get("time steps"));
    TEST_EQUALITY(tNumCells, tTimeStepWS.extent(0));
    auto tNumNodesPerCell = PhysicsT::mNumNodesPerCell;
    TEST_EQUALITY(tNumNodesPerCell, tTimeStepWS.extent(1));
    auto tHostTimeStepWS = Kokkos::create_mirror(tTimeStepWS);
    Kokkos::deep_copy(tHostTimeStepWS, tTimeStepWS);
    for (decltype(tNumCells) tCell = 0; tCell < tNumCells; tCell++)
    {
        for (decltype(tNumNodesPerCell) tDof = 0; tDof < tNumNodesPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(4.0, tHostTimeStepWS(tCell, tDof), tTol);
        }
    }

    // test artificial compressibility results
    auto tArtCompressWS = Plato::metadata<Plato::ScalarMultiVector>(tWorkSets.get("artificial compressibility"));
    TEST_EQUALITY(tNumCells, tArtCompressWS.extent(0));
    TEST_EQUALITY(tNumNodesPerCell, tArtCompressWS.extent(1));
    auto tHostArtCompressWS = Kokkos::create_mirror(tArtCompressWS);
    Kokkos::deep_copy(tHostArtCompressWS, tArtCompressWS);
    for (decltype(tNumCells) tCell = 0; tCell < tNumCells; tCell++)
    {
        for (decltype(tNumNodesPerCell) tDof = 0; tDof < tNumNodesPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(5.0, tHostArtCompressWS(tCell, tDof), tTol);
        }
    }

    // test controls results
    auto tControlWS = Plato::metadata<Plato::ScalarMultiVectorT<ResidualEvalT::ControlScalarType>>(tWorkSets.get("control"));
    TEST_EQUALITY(tNumCells, tControlWS.extent(0));
    TEST_EQUALITY(tNumNodesPerCell, tControlWS.extent(1));
    auto tHostControlWS = Kokkos::create_mirror(tControlWS);
    Kokkos::deep_copy(tHostControlWS, tControlWS);
    for (decltype(tNumCells) tCell = 0; tCell < tNumCells; tCell++)
    {
        for (decltype(tNumNodesPerCell) tDof = 0; tDof < tNumNodesPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(0.5, tHostControlWS(tCell, tDof), tTol);
        }
    }

    // test prescribed velocity results
    auto tVelBCsWS = Plato::metadata<Plato::ScalarMultiVector>(tWorkSets.get("prescribed velocity"));
    TEST_EQUALITY(tNumCells, tVelBCsWS.extent(0));
    TEST_EQUALITY(tNumVelDofsPerCell, tVelBCsWS.extent(1));
    auto tHostVelBCsWS = Kokkos::create_mirror(tVelBCsWS);
    Kokkos::deep_copy(tHostVelBCsWS, tVelBCsWS);
    for (decltype(tNumCells) tCell = 0; tCell < tNumCells; tCell++)
    {
        for (decltype(tNumVelDofsPerCell) tDof = 0; tDof < tNumVelDofsPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(6.0, tHostVelBCsWS(tCell, tDof), tTol);
        }
    }

    // test configuration results
    auto tConfigWS = Plato::metadata<Plato::ScalarArray3DT<ResidualEvalT::ConfigScalarType>>(tWorkSets.get("configuration"));
    TEST_EQUALITY(tNumCells, tConfigWS.extent(0));
    TEST_EQUALITY(tNumNodesPerCell, tConfigWS.extent(1));
    auto tNumConfigDofsPerNode = PhysicsT::mNumConfigDofsPerNode;
    TEST_EQUALITY(tNumConfigDofsPerNode, tConfigWS.extent(2));
    auto tHostConfigWS = Kokkos::create_mirror(tConfigWS);
    Kokkos::deep_copy(tHostConfigWS, tConfigWS);
    Plato::ScalarArray3D tGoldConfigWS("gold configuration", tNumCells, tNumNodesPerCell, tNumConfigDofsPerNode);
    auto tHostGoldConfigWS = Kokkos::create_mirror(tGoldConfigWS);
    tHostGoldConfigWS(0,0,0) = 0; tHostGoldConfigWS(0,1,0) = 1; tHostGoldConfigWS(0,2,0) = 1;
    tHostGoldConfigWS(1,0,0) = 1; tHostGoldConfigWS(1,1,0) = 0; tHostGoldConfigWS(1,2,0) = 0;
    tHostGoldConfigWS(0,0,1) = 0; tHostGoldConfigWS(0,1,1) = 0; tHostGoldConfigWS(0,2,1) = 1;
    tHostGoldConfigWS(1,0,1) = 1; tHostGoldConfigWS(1,1,1) = 1; tHostGoldConfigWS(1,2,1) = 0;
    for (decltype(tNumCells) tCell = 0; tCell < tNumCells; tCell++)
    {
        for (decltype(tNumNodesPerCell) tNode = 0; tNode < tNumNodesPerCell; tNode++)
        {
            for (decltype(tNumConfigDofsPerNode) tDof = 0; tDof < tNumConfigDofsPerNode; tDof++)
            {
                TEST_FLOATING_EQUALITY(tHostGoldConfigWS(tCell, tNode, tDof), tHostConfigWS(tCell, tNode, tDof), tTol);
            }
        }
    }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, BuildVectorFunctionWorksetsTwo)
{
    constexpr Plato::OrdinalType tNumSpaceDim = 2;
    using PhysicsT = Plato::IncompressibleFluids<tNumSpaceDim>;
    using ResidualEvalT = Plato::FluidMechanics::Evaluation<PhysicsT::SimplexT>::Residual;

    // set current state
    Plato::Variables tPrimal;
    auto tNumCells = 2;
    auto tNumNodes = 4;
    auto tNumVelDofs = tNumNodes * tNumSpaceDim;
    Plato::ScalarVector tControls("controls", tNumNodes);
    Plato::blas1::fill(0.5, tControls);
    Plato::ScalarVector tCurPred("current predictor", tNumVelDofs);
    Plato::blas1::fill(0.1, tCurPred);
    tPrimal.vector("current predictor", tCurPred);
    Plato::ScalarVector tCurVel("current velocity", tNumVelDofs);
    Plato::blas1::fill(1.0, tCurVel);
    tPrimal.vector("current velocity", tCurVel);
    Plato::ScalarVector tCurPress("current pressure", tNumNodes);
    Plato::blas1::fill(2.0, tCurPress);
    tPrimal.vector("current pressure", tCurPress);
    Plato::ScalarVector tCurTemp("current temperature", tNumNodes);
    Plato::blas1::fill(3.0, tCurTemp);
    tPrimal.vector("current temperature", tCurTemp);
    Plato::ScalarVector tPrevVel("previous velocity", tNumVelDofs);
    Plato::blas1::fill(0.8, tPrevVel);
    tPrimal.vector("previous velocity", tPrevVel);
    Plato::ScalarVector tPrevPress("previous pressure", tNumNodes);
    Plato::blas1::fill(1.8, tPrevPress);
    tPrimal.vector("previous pressure", tPrevPress);
    Plato::ScalarVector tPrevTemp("previous temperature", tNumNodes);
    Plato::blas1::fill(2.8, tPrevTemp);
    tPrimal.vector("previous temperature", tPrevTemp);
    Plato::ScalarVector tTimeSteps("time steps", tNumNodes);
    Plato::blas1::fill(4.0, tTimeSteps);
    tPrimal.vector("time steps", tTimeSteps);
    Plato::ScalarVector tVelBCs("prescribed velocity", tNumVelDofs);
    Plato::blas1::fill(6.0, tVelBCs);
    tPrimal.vector("prescribed velocity", tVelBCs);

    // set ordinal maps;
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);
    Plato::LocalOrdinalMaps<PhysicsT> tOrdinalMaps(*tMesh);

    // call build_vector_function_worksets
    Plato::WorkSets tWorkSets;
    Plato::FluidMechanics::build_vector_function_worksets<ResidualEvalT>
        (tNumCells, tControls, tPrimal, tOrdinalMaps, tWorkSets);
    TEST_EQUALITY(tWorkSets.defined("artifical compressibility"), false);
}


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, BuildScalarFunctionWorksets)
{
    constexpr Plato::OrdinalType tNumSpaceDim = 2;
    using PhysicsT = Plato::IncompressibleFluids<tNumSpaceDim>;
    using ResidualEvalT = Plato::FluidMechanics::Evaluation<PhysicsT::SimplexT>::Residual;

    // set current state
    Plato::Variables tPrimal;
    auto tNumCells = 2;
    auto tNumNodes = 4;
    auto tNumVelDofs = tNumNodes * tNumSpaceDim;
    Plato::ScalarVector tControls("controls", tNumNodes);
    Plato::blas1::fill(0.5, tControls);
    Plato::ScalarVector tCurVel("current velocity", tNumVelDofs);
    Plato::blas1::fill(1.0, tCurVel);
    tPrimal.vector("current velocity", tCurVel);
    Plato::ScalarVector tCurPress("current pressure", tNumNodes);
    Plato::blas1::fill(2.0, tCurPress);
    tPrimal.vector("current pressure", tCurPress);
    Plato::ScalarVector tCurTemp("current temperature", tNumNodes);
    Plato::blas1::fill(3.0, tCurTemp);
    tPrimal.vector("current temperature", tCurTemp);
    Plato::ScalarVector tTimeSteps("time steps", tNumNodes);
    Plato::blas1::fill(4.0, tTimeSteps);
    tPrimal.vector("time steps", tTimeSteps);

    // set ordinal maps;
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);
    Plato::LocalOrdinalMaps<PhysicsT> tOrdinalMaps(*tMesh);

    // call build_scalar_function_worksets
    Plato::WorkSets tWorkSets;
    Plato::FluidMechanics::build_scalar_function_worksets<ResidualEvalT>
        (tNumCells, tControls, tPrimal, tOrdinalMaps, tWorkSets);

    // test current velocity results
    auto tCurVelWS = Plato::metadata<Plato::ScalarMultiVectorT<ResidualEvalT::CurrentMomentumScalarType>>(tWorkSets.get("current velocity"));
    TEST_EQUALITY(tNumCells, tCurVelWS.extent(0));
    auto tNumVelDofsPerCell = PhysicsT::mNumMomentumDofsPerCell;
    TEST_EQUALITY(tNumVelDofsPerCell, tCurVelWS.extent(1));
    auto tHostCurVelWS = Kokkos::create_mirror(tCurVelWS);
    Kokkos::deep_copy(tHostCurVelWS, tCurVelWS);
    const Plato::Scalar tTol = 1e-6;
    for (decltype(tNumCells) tCell = 0; tCell < tNumCells; tCell++)
    {
        for (decltype(tNumVelDofsPerCell) tDof = 0; tDof < tNumVelDofsPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(1.0, tHostCurVelWS(tCell, tDof), tTol);
        }
    }

    // test current pressure results
    auto tCurPressWS = Plato::metadata<Plato::ScalarMultiVectorT<ResidualEvalT::CurrentMassScalarType>>(tWorkSets.get("current pressure"));
    TEST_EQUALITY(tNumCells, tCurPressWS.extent(0));
    auto tNumPressDofsPerCell = PhysicsT::mNumMassDofsPerCell;
    TEST_EQUALITY(tNumPressDofsPerCell, tCurPressWS.extent(1));
    auto tHostCurPressWS = Kokkos::create_mirror(tCurPressWS);
    Kokkos::deep_copy(tHostCurPressWS, tCurPressWS);
    for (decltype(tNumCells) tCell = 0; tCell < tNumCells; tCell++)
    {
        for (decltype(tNumPressDofsPerCell) tDof = 0; tDof < tNumPressDofsPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(2.0, tHostCurPressWS(tCell, tDof), tTol);
        }
    }

    // test current temperature results
    auto tCurTempWS = Plato::metadata<Plato::ScalarMultiVectorT<ResidualEvalT::CurrentEnergyScalarType>>(tWorkSets.get("current temperature"));
    TEST_EQUALITY(tNumCells, tCurTempWS.extent(0));
    auto tNumTempDofsPerCell = PhysicsT::mNumEnergyDofsPerCell;
    TEST_EQUALITY(tNumTempDofsPerCell, tCurTempWS.extent(1));
    auto tHostCurTempWS = Kokkos::create_mirror(tCurTempWS);
    Kokkos::deep_copy(tHostCurTempWS, tCurTempWS);
    for (decltype(tNumCells) tCell = 0; tCell < tNumCells; tCell++)
    {
        for (decltype(tNumTempDofsPerCell) tDof = 0; tDof < tNumTempDofsPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(3.0, tHostCurTempWS(tCell, tDof), tTol);
        }
    }

    // test time steps results
    auto tTimeStepWS = Plato::metadata<Plato::ScalarMultiVector>(tWorkSets.get("time steps"));
    TEST_EQUALITY(tNumCells, tTimeStepWS.extent(0));
    auto tNumNodesPerCell = PhysicsT::mNumNodesPerCell;
    TEST_EQUALITY(tNumNodesPerCell, tTimeStepWS.extent(1));
    auto tHostTimeStepWS = Kokkos::create_mirror(tTimeStepWS);
    Kokkos::deep_copy(tHostTimeStepWS, tTimeStepWS);
    for (decltype(tNumCells) tCell = 0; tCell < tNumCells; tCell++)
    {
        for (decltype(tNumNodesPerCell) tDof = 0; tDof < tNumNodesPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(4.0, tHostTimeStepWS(tCell, tDof), tTol);
        }
    }

    // test controls results
    auto tControlWS = Plato::metadata<Plato::ScalarMultiVectorT<ResidualEvalT::ControlScalarType>>(tWorkSets.get("control"));
    TEST_EQUALITY(tNumCells, tControlWS.extent(0));
    TEST_EQUALITY(tNumNodesPerCell, tControlWS.extent(1));
    auto tHostControlWS = Kokkos::create_mirror(tControlWS);
    Kokkos::deep_copy(tHostControlWS, tControlWS);
    for (decltype(tNumCells) tCell = 0; tCell < tNumCells; tCell++)
    {
        for (decltype(tNumNodesPerCell) tDof = 0; tDof < tNumNodesPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(0.5, tHostControlWS(tCell, tDof), tTol);
        }
    }

    // test configuration results
    auto tConfigWS = Plato::metadata<Plato::ScalarArray3DT<ResidualEvalT::ConfigScalarType>>(tWorkSets.get("configuration"));
    TEST_EQUALITY(tNumCells, tConfigWS.extent(0));
    TEST_EQUALITY(tNumNodesPerCell, tConfigWS.extent(1));
    auto tNumConfigDofsPerNode = PhysicsT::mNumConfigDofsPerNode;
    TEST_EQUALITY(tNumConfigDofsPerNode, tConfigWS.extent(2));
    auto tHostConfigWS = Kokkos::create_mirror(tConfigWS);
    Kokkos::deep_copy(tHostConfigWS, tConfigWS);
    Plato::ScalarArray3D tGoldConfigWS("gold configuration", tNumCells, tNumNodesPerCell, tNumConfigDofsPerNode);
    auto tHostGoldConfigWS = Kokkos::create_mirror(tGoldConfigWS);
    tHostGoldConfigWS(0,0,0) = 0; tHostGoldConfigWS(0,1,0) = 1; tHostGoldConfigWS(0,2,0) = 1;
    tHostGoldConfigWS(1,0,0) = 1; tHostGoldConfigWS(1,1,0) = 0; tHostGoldConfigWS(1,2,0) = 0;
    tHostGoldConfigWS(0,0,1) = 0; tHostGoldConfigWS(0,1,1) = 0; tHostGoldConfigWS(0,2,1) = 1;
    tHostGoldConfigWS(1,0,1) = 1; tHostGoldConfigWS(1,1,1) = 1; tHostGoldConfigWS(1,2,1) = 0;
    for (decltype(tNumCells) tCell = 0; tCell < tNumCells; tCell++)
    {
        for (decltype(tNumNodesPerCell) tNode = 0; tNode < tNumNodesPerCell; tNode++)
        {
            for (decltype(tNumConfigDofsPerNode) tDof = 0; tDof < tNumConfigDofsPerNode; tDof++)
            {
                TEST_FLOATING_EQUALITY(tHostGoldConfigWS(tCell, tNode, tDof), tHostConfigWS(tCell, tNode, tDof), tTol);
            }
        }
    }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ParseArray)
{
    Teuchos::RCP<Teuchos::ParameterList> tParams =
        Teuchos::getParametersFromXmlString(
            "<ParameterList  name='Criteria'>"
            "  <Parameter  name='Type'         type='string'         value='Weighted Sum'/>"
            "  <Parameter  name='Functions'    type='Array(string)'  value='{My Inlet Pressure, My Outlet Pressure}'/>"
            "  <Parameter  name='Weights'      type='Array(double)'  value='{1.0,-1.0}'/>"
            "  <ParameterList  name='My Inlet Pressure'>"
            "    <Parameter  name='Type'                   type='string'           value='Scalar Function'/>"
            "    <Parameter  name='Scalar Function Type'   type='string'           value='Average Surface Pressure'/>"
            "    <Parameter  name='Sides'                  type='Array(string)'    value='{ss_1}'/>"
            "  </ParameterList>"
            "  <ParameterList  name='My Outlet Pressure'>"
            "    <Parameter  name='Type'                   type='string'           value='Scalar Function'/>"
            "    <Parameter  name='Scalar Function Type'   type='string'           value='Average Surface Pressure'/>"
            "    <Parameter  name='Sides'                  type='Array(string)'    value='{ss_2}'/>"
            "  </ParameterList>"
            "</ParameterList>"
            );
    auto tNames = Plato::parse_array<std::string>("Functions", tParams.operator*());

    std::vector<std::string> tGoldNames = {"My Inlet Pressure", "My Outlet Pressure"};
    for(auto& tName : tNames)
    {
        auto tIndex = &tName - &tNames[0];
        TEST_EQUALITY(tGoldNames[tIndex], tName);
    }

    auto tWeights = Plato::parse_array<Plato::Scalar>("Weights", *tParams);
    std::vector<Plato::Scalar> tGoldWeights = {1.0, -1.0};
    for(auto& tWeight : tWeights)
    {
        auto tIndex = &tWeight - &tWeights[0];
        TEST_EQUALITY(tGoldWeights[tIndex], tWeight);
    }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, WorkStes)
{
    Plato::WorkSets tWorkSets;

    Plato::OrdinalType tNumCells = 1;
    Plato::OrdinalType tNumVelDofs = 12;
    Plato::ScalarMultiVector tVelWS("velocity", tNumCells, tNumVelDofs);
    Plato::blas2::fill(1.0, tVelWS);
    auto tVelPtr = std::make_shared<Plato::MetaData<Plato::ScalarMultiVector>>( tVelWS );
    tWorkSets.set("velocity", tVelPtr);

    Plato::OrdinalType tNumPressDofs = 4;
    Plato::ScalarMultiVector tPressWS("pressure", tNumCells, tNumPressDofs);
    Plato::blas2::fill(2.0, tPressWS);
    auto tPressPtr = std::make_shared<Plato::MetaData<Plato::ScalarMultiVector>>( tPressWS );
    tWorkSets.set("pressure", tPressPtr);

    // TEST VALUES
    tVelWS = Plato::metadata<Plato::ScalarMultiVector>(tWorkSets.get("velocity"));
    TEST_EQUALITY(tNumCells, tVelWS.extent(0));
    TEST_EQUALITY(tNumVelDofs, tVelWS.extent(1));
    auto tHostVelWS = Kokkos::create_mirror(tVelWS);
    Kokkos::deep_copy(tHostVelWS, tVelWS);
    const Plato::Scalar tTol = 1e-6;
    for(decltype(tNumVelDofs) tIndex = 0; tIndex < tNumVelDofs; tIndex++)
    {
        TEST_FLOATING_EQUALITY(1.0, tHostVelWS(0, tIndex), tTol);
    }

    tPressWS = Plato::metadata<Plato::ScalarMultiVector>(tWorkSets.get("pressure"));
    TEST_EQUALITY(tNumCells, tPressWS.extent(0));
    TEST_EQUALITY(tNumPressDofs, tPressWS.extent(1));
    auto tHostPressWS = Kokkos::create_mirror(tPressWS);
    Kokkos::deep_copy(tHostPressWS, tPressWS);
    for(decltype(tNumPressDofs) tIndex = 0; tIndex < tNumPressDofs; tIndex++)
    {
        TEST_FLOATING_EQUALITY(2.0, tHostPressWS(0, tIndex), tTol);
    }

    // TEST TAGS
    auto tTags = tWorkSets.tags();
    std::vector<std::string> tGoldTags = {"velocity", "pressure"};
    for(auto& tTag : tTags)
    {
        auto tGoldItr = std::find(tGoldTags.begin(), tGoldTags.end(), tTag);
        if(tGoldItr != tGoldTags.end())
        {
            TEST_EQUALITY(tGoldItr.operator*(), tTag);
        }
        else
        {
            TEST_EQUALITY("failed", tTag);
        }
    }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, LocalOrdinalMaps)
{
    constexpr Plato::OrdinalType tNumSpaceDim = 3;
    using PhysicsT = Plato::MomentumConservation<tNumSpaceDim>;
    auto tMesh = PlatoUtestHelpers::build_3d_box_mesh(1.0, 1.0, 1.0, 1, 1, 1);
    Plato::LocalOrdinalMaps<PhysicsT> tLocalOrdinalMaps(tMesh.operator*());

    auto tNumCells = tMesh->nelems();
    Plato::ScalarArray3D tCoords("coordinates", tNumCells, PhysicsT::mNumNodesPerCell, tNumSpaceDim);
    Plato::ScalarMultiVector tControlOrdinals("control", tNumCells, PhysicsT::mNumNodesPerCell);
    Plato::ScalarMultiVector tScalarOrdinals("scalar field", tNumCells, PhysicsT::mNumNodesPerCell);
    Plato::ScalarArray3D tVectorOrdinals("vector field", tNumCells, PhysicsT::mNumNodesPerCell, PhysicsT::mNumMomentumDofsPerNode);

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        for(Plato::OrdinalType tNode = 0; tNode < PhysicsT::mNumNodesPerCell; tNode++)
        {
            for(Plato::OrdinalType tDim = 0; tDim < PhysicsT::mNumSpatialDims; tDim++)
            {
                tCoords(aCellOrdinal, tNode, tDim) = tLocalOrdinalMaps.mNodeCoordinate(aCellOrdinal, tNode, tDim);
                tVectorOrdinals(aCellOrdinal, tNode, tDim) = tLocalOrdinalMaps.mVectorStateOrdinalMap(aCellOrdinal, tNode, tDim);
            }
        }

        for(Plato::OrdinalType tNode = 0; tNode < PhysicsT::mNumNodesPerCell; tNode++)
        {
            for(Plato::OrdinalType tDim = 0; tDim < PhysicsT::mNumControlDofsPerNode; tDim++)
            {
                tControlOrdinals(aCellOrdinal, tNode) = tLocalOrdinalMaps.mControlOrdinalMap(aCellOrdinal, tNode, tDim);
                tScalarOrdinals(aCellOrdinal, tNode) = tLocalOrdinalMaps.mScalarStateOrdinalMap(aCellOrdinal, tNode, tDim);
            }
        }

    },"test");

    // TEST 3D ARRAYS
    Plato::ScalarArray3D tGoldCoords("coordinates", tNumCells, PhysicsT::mNumNodesPerCell, tNumSpaceDim);
    auto tHostGoldCoords = Kokkos::create_mirror(tGoldCoords);
    tHostGoldCoords(0,0,0) = 0; tHostGoldCoords(0,1,0) = 1; tHostGoldCoords(0,2,0) = 0; tHostGoldCoords(0,3,0) = 1;
    tHostGoldCoords(1,0,0) = 0; tHostGoldCoords(1,1,0) = 0; tHostGoldCoords(1,2,0) = 0; tHostGoldCoords(1,3,0) = 1;
    tHostGoldCoords(2,0,0) = 0; tHostGoldCoords(2,1,0) = 0; tHostGoldCoords(2,2,0) = 0; tHostGoldCoords(2,3,0) = 1;
    tHostGoldCoords(3,0,0) = 0; tHostGoldCoords(3,1,0) = 1; tHostGoldCoords(3,2,0) = 1; tHostGoldCoords(3,3,0) = 0;
    tHostGoldCoords(4,0,0) = 1; tHostGoldCoords(4,1,0) = 1; tHostGoldCoords(4,2,0) = 1; tHostGoldCoords(4,3,0) = 0;
    tHostGoldCoords(5,0,0) = 1; tHostGoldCoords(5,1,0) = 1; tHostGoldCoords(5,2,0) = 1; tHostGoldCoords(5,3,0) = 0;
    tHostGoldCoords(0,0,1) = 0; tHostGoldCoords(0,1,1) = 1; tHostGoldCoords(0,2,1) = 1; tHostGoldCoords(0,3,1) = 1;
    tHostGoldCoords(1,0,1) = 0; tHostGoldCoords(1,1,1) = 1; tHostGoldCoords(1,2,1) = 1; tHostGoldCoords(1,3,1) = 1;
    tHostGoldCoords(2,0,1) = 0; tHostGoldCoords(2,1,1) = 1; tHostGoldCoords(2,2,1) = 0; tHostGoldCoords(2,3,1) = 1;
    tHostGoldCoords(3,0,1) = 0; tHostGoldCoords(3,1,1) = 0; tHostGoldCoords(3,2,1) = 1; tHostGoldCoords(3,3,1) = 0;
    tHostGoldCoords(4,0,1) = 0; tHostGoldCoords(4,1,1) = 0; tHostGoldCoords(4,2,1) = 1; tHostGoldCoords(4,3,1) = 0;
    tHostGoldCoords(5,0,1) = 0; tHostGoldCoords(5,1,1) = 1; tHostGoldCoords(5,2,1) = 1; tHostGoldCoords(5,3,1) = 0;
    tHostGoldCoords(0,0,2) = 0; tHostGoldCoords(0,1,2) = 0; tHostGoldCoords(0,2,2) = 0; tHostGoldCoords(0,3,2) = 1;
    tHostGoldCoords(1,0,2) = 0; tHostGoldCoords(1,1,2) = 0; tHostGoldCoords(1,2,2) = 1; tHostGoldCoords(1,3,2) = 1;
    tHostGoldCoords(2,0,2) = 0; tHostGoldCoords(2,1,2) = 1; tHostGoldCoords(2,2,2) = 1; tHostGoldCoords(2,3,2) = 1;
    tHostGoldCoords(3,0,2) = 0; tHostGoldCoords(3,1,2) = 1; tHostGoldCoords(3,2,2) = 1; tHostGoldCoords(3,3,2) = 1;
    tHostGoldCoords(4,0,2) = 0; tHostGoldCoords(4,1,2) = 1; tHostGoldCoords(4,2,2) = 1; tHostGoldCoords(4,3,2) = 0;
    tHostGoldCoords(5,0,2) = 0; tHostGoldCoords(5,1,2) = 1; tHostGoldCoords(5,2,2) = 0; tHostGoldCoords(5,3,2) = 0;
    auto tHostCoords = Kokkos::create_mirror(tCoords);
    Kokkos::deep_copy(tHostCoords, tCoords);

    Plato::ScalarArray3D tGoldVectorOrdinals("vector field", tNumCells, PhysicsT::mNumNodesPerCell, PhysicsT::mNumMomentumDofsPerNode);
    auto tHostGoldVecOrdinals = Kokkos::create_mirror(tGoldVectorOrdinals);
    tHostGoldVecOrdinals(0,0,0) = 0;  tHostGoldVecOrdinals(0,1,0) = 12; tHostGoldVecOrdinals(0,2,0) = 9;  tHostGoldVecOrdinals(0,3,0) = 15;
    tHostGoldVecOrdinals(1,0,0) = 0;  tHostGoldVecOrdinals(1,1,0) = 9;  tHostGoldVecOrdinals(1,2,0) = 6;  tHostGoldVecOrdinals(1,3,0) = 15;
    tHostGoldVecOrdinals(2,0,0) = 0;  tHostGoldVecOrdinals(2,1,0) = 6;  tHostGoldVecOrdinals(2,2,0) = 3;  tHostGoldVecOrdinals(2,3,0) = 15;
    tHostGoldVecOrdinals(3,0,0) = 0;  tHostGoldVecOrdinals(3,1,0) = 18; tHostGoldVecOrdinals(3,2,0) = 15; tHostGoldVecOrdinals(3,3,0) = 3;
    tHostGoldVecOrdinals(4,0,0) = 21; tHostGoldVecOrdinals(4,1,0) = 18; tHostGoldVecOrdinals(4,2,0) = 15; tHostGoldVecOrdinals(4,3,0) = 0;
    tHostGoldVecOrdinals(5,0,0) = 21; tHostGoldVecOrdinals(5,1,0) = 15; tHostGoldVecOrdinals(5,2,0) = 12; tHostGoldVecOrdinals(5,3,0) = 0;
    tHostGoldVecOrdinals(0,0,1) = 1;  tHostGoldVecOrdinals(0,1,1) = 13; tHostGoldVecOrdinals(0,2,1) = 10; tHostGoldVecOrdinals(0,3,1) = 16;
    tHostGoldVecOrdinals(1,0,1) = 1;  tHostGoldVecOrdinals(1,1,1) = 10; tHostGoldVecOrdinals(1,2,1) = 7;  tHostGoldVecOrdinals(1,3,1) = 16;
    tHostGoldVecOrdinals(2,0,1) = 1;  tHostGoldVecOrdinals(2,1,1) = 7;  tHostGoldVecOrdinals(2,2,1) = 4;  tHostGoldVecOrdinals(2,3,1) = 16;
    tHostGoldVecOrdinals(3,0,1) = 1;  tHostGoldVecOrdinals(3,1,1) = 19; tHostGoldVecOrdinals(3,2,1) = 16; tHostGoldVecOrdinals(3,3,1) = 4;
    tHostGoldVecOrdinals(4,0,1) = 22; tHostGoldVecOrdinals(4,1,1) = 19; tHostGoldVecOrdinals(4,2,1) = 16; tHostGoldVecOrdinals(4,3,1) = 1;
    tHostGoldVecOrdinals(5,0,1) = 22; tHostGoldVecOrdinals(5,1,1) = 16; tHostGoldVecOrdinals(5,2,1) = 13; tHostGoldVecOrdinals(5,3,1) = 1;
    tHostGoldVecOrdinals(0,0,2) = 2;  tHostGoldVecOrdinals(0,1,2) = 14; tHostGoldVecOrdinals(0,2,2) = 11; tHostGoldVecOrdinals(0,3,2) = 17;
    tHostGoldVecOrdinals(1,0,2) = 2;  tHostGoldVecOrdinals(1,1,2) = 11; tHostGoldVecOrdinals(1,2,2) = 8;  tHostGoldVecOrdinals(1,3,2) = 17;
    tHostGoldVecOrdinals(2,0,2) = 2;  tHostGoldVecOrdinals(2,1,2) = 8;  tHostGoldVecOrdinals(2,2,2) = 5;  tHostGoldVecOrdinals(2,3,2) = 17;
    tHostGoldVecOrdinals(3,0,2) = 2;  tHostGoldVecOrdinals(3,1,2) = 20; tHostGoldVecOrdinals(3,2,2) = 17; tHostGoldVecOrdinals(3,3,2) = 5;
    tHostGoldVecOrdinals(4,0,2) = 23; tHostGoldVecOrdinals(4,1,2) = 20; tHostGoldVecOrdinals(4,2,2) = 17; tHostGoldVecOrdinals(4,3,2) = 2;
    tHostGoldVecOrdinals(5,0,2) = 23; tHostGoldVecOrdinals(5,1,2) = 17; tHostGoldVecOrdinals(5,2,2) = 14; tHostGoldVecOrdinals(5,3,2) = 2;
    auto tHostVectorOrdinals = Kokkos::create_mirror(tVectorOrdinals);
    Kokkos::deep_copy(tHostVectorOrdinals, tVectorOrdinals);

    auto tTol = 1e-6;
    for(Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++)
    {
        for(Plato::OrdinalType tNode = 0; tNode < PhysicsT::mNumNodesPerCell; tNode++)
        {
            for(Plato::OrdinalType tDim = 0; tDim < PhysicsT::mNumSpatialDims; tDim++)
            {
                TEST_FLOATING_EQUALITY(tHostGoldCoords(tCell, tNode, tDim), tHostCoords(tCell, tNode, tDim), tTol);
                TEST_FLOATING_EQUALITY(tHostGoldVecOrdinals(tCell, tNode, tDim), tHostVectorOrdinals(tCell, tNode, tDim), tTol);
            }
        }
    }

    // TEST 2D ARRAYS
    Plato::ScalarMultiVector tGoldControlOrdinals("control", tNumCells, PhysicsT::mNumNodesPerCell);
    auto tHostGoldControlOrdinals = Kokkos::create_mirror(tGoldControlOrdinals);
    tHostGoldControlOrdinals(0,0) = 0; tHostGoldControlOrdinals(0,1) = 4; tHostGoldControlOrdinals(0,2) = 3; tHostGoldControlOrdinals(0,3) = 5;
    tHostGoldControlOrdinals(1,0) = 0; tHostGoldControlOrdinals(1,1) = 3; tHostGoldControlOrdinals(1,2) = 2; tHostGoldControlOrdinals(1,3) = 5;
    tHostGoldControlOrdinals(2,0) = 0; tHostGoldControlOrdinals(2,1) = 2; tHostGoldControlOrdinals(2,2) = 1; tHostGoldControlOrdinals(2,3) = 5;
    tHostGoldControlOrdinals(3,0) = 0; tHostGoldControlOrdinals(3,1) = 6; tHostGoldControlOrdinals(3,2) = 5; tHostGoldControlOrdinals(3,3) = 1;
    tHostGoldControlOrdinals(4,0) = 7; tHostGoldControlOrdinals(4,1) = 6; tHostGoldControlOrdinals(4,2) = 5; tHostGoldControlOrdinals(4,3) = 0;
    tHostGoldControlOrdinals(5,0) = 7; tHostGoldControlOrdinals(5,1) = 5; tHostGoldControlOrdinals(5,2) = 4; tHostGoldControlOrdinals(5,3) = 0;
    auto tHostControlOrdinals = Kokkos::create_mirror(tControlOrdinals);
    Kokkos::deep_copy(tHostControlOrdinals, tControlOrdinals);

    Plato::ScalarMultiVector tGoldScalarOrdinals("scalar field", tNumCells, PhysicsT::mNumNodesPerCell);
    auto tHostGoldScalarOrdinals = Kokkos::create_mirror(tGoldScalarOrdinals);
    tHostGoldScalarOrdinals(0,0) = 0; tHostGoldScalarOrdinals(0,1) = 4; tHostGoldScalarOrdinals(0,2) = 3; tHostGoldScalarOrdinals(0,3) = 5;
    tHostGoldScalarOrdinals(1,0) = 0; tHostGoldScalarOrdinals(1,1) = 3; tHostGoldScalarOrdinals(1,2) = 2; tHostGoldScalarOrdinals(1,3) = 5;
    tHostGoldScalarOrdinals(2,0) = 0; tHostGoldScalarOrdinals(2,1) = 2; tHostGoldScalarOrdinals(2,2) = 1; tHostGoldScalarOrdinals(2,3) = 5;
    tHostGoldScalarOrdinals(3,0) = 0; tHostGoldScalarOrdinals(3,1) = 6; tHostGoldScalarOrdinals(3,2) = 5; tHostGoldScalarOrdinals(3,3) = 1;
    tHostGoldScalarOrdinals(4,0) = 7; tHostGoldScalarOrdinals(4,1) = 6; tHostGoldScalarOrdinals(4,2) = 5; tHostGoldScalarOrdinals(4,3) = 0;
    tHostGoldScalarOrdinals(5,0) = 7; tHostGoldScalarOrdinals(5,1) = 5; tHostGoldScalarOrdinals(5,2) = 4; tHostGoldScalarOrdinals(5,3) = 0;
    auto tHostScalarOrdinals = Kokkos::create_mirror(tScalarOrdinals);
    Kokkos::deep_copy(tHostScalarOrdinals, tScalarOrdinals);

    for(Plato::OrdinalType tNode = 0; tNode < PhysicsT::mNumNodesPerCell; tNode++)
    {
        for(Plato::OrdinalType tDim = 0; tDim < PhysicsT::mNumSpatialDims; tDim++)
        {
            TEST_FLOATING_EQUALITY(tHostGoldControlOrdinals(tNode, tDim), tHostControlOrdinals(tNode, tDim), tTol);
            TEST_FLOATING_EQUALITY(tHostGoldScalarOrdinals(tNode, tDim), tHostScalarOrdinals(tNode, tDim), tTol);
        }
    }
}

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
    auto tScalarOutput = Plato::parse_parameter<Plato::Scalar>("Prandtl", "Dimensionless Properties", tParams.operator*());
    auto tTolerance = 1e-6;
    TEST_FLOATING_EQUALITY(tScalarOutput, 2.1, tTolerance);

    // Darcy #
    tScalarOutput = Plato::parse_parameter<Plato::Scalar>("Darcy", "Dimensionless Properties", tParams.operator*());
    TEST_FLOATING_EQUALITY(tScalarOutput, 2.2, tTolerance);

    // Grashof #
    auto tArrayOutput = Plato::parse_parameter<Teuchos::Array<Plato::Scalar>>("Grashof", "Dimensionless Properties", tParams.operator*());
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
    Kokkos::deep_copy(tHostVel, tVel);
    tHostGoldVel  = Kokkos::create_mirror(tGoldVel);
    Kokkos::deep_copy(tHostGoldVel, tGoldVel);
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
    Kokkos::deep_copy(tHostPress, tPress);
    tHostGoldPress  = Kokkos::create_mirror(tGoldPress);
    Kokkos::deep_copy(tHostGoldPress, tGoldPress);
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
    Kokkos::deep_copy(tHostTemp, tTemp);
    tHostGoldTemp  = Kokkos::create_mirror(tGoldTemp);
    Kokkos::deep_copy(tHostGoldTemp, tGoldTemp);
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
    Plato::Variables tStates;
    TEST_COMPARE(tStates.isScalarMapEmpty(), ==, true);
    TEST_COMPARE(tStates.isVectorMapEmpty(), ==, true);

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
    TEST_COMPARE(tStates.defined("velocity"), ==, true);
    TEST_COMPARE(tStates.defined("temperature"), ==, false);

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

}
