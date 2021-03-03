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
#include "Plato_Solve.hpp"
#include "SpatialModel.hpp"
#include "EssentialBCs.hpp"
#include "ProjectToNode.hpp"
#include "PlatoUtilities.hpp"
#include "OmegaHUtilities.hpp"
#include "SimplexFadTypes.hpp"
#include "ApplyConstraints.hpp"
#include "PlatoMathHelpers.hpp"
#include "PlatoAbstractProblem.hpp"
#include "Plato_TopOptFunctors.hpp"
#include "InterpolateFromNodal.hpp"
#include "SurfaceIntegralUtilities.hpp"
#include "LinearTetCubRuleDegreeOne.hpp"

#include "alg/PlatoSolverFactory.hpp"

#include "PlatoTestHelpers.hpp"

namespace Plato
{

namespace blas2
{

/******************************************************************************//**
 * \tparam LengthI    number of elements in the i-th index direction
 * \tparam LengthJ    number of elements in the j-th index direction
 * \tparam ScalarT    POD type
 * \tparam AViewTypeT view type
 * \tparam BViewTypeT view type
 *
 * \fn device_type inline void scale
 *
 * \brief Scale all the elements by input scalar value
 *
 * \param [in]  aCellOrdinal cell/element ordinal
 * \param [in]  aScalar      input scalar
 * \param [in]  aInputWS     input 3D scalar view
 * \param [out] aOutputWS    output 3D scalar view
**********************************************************************************/
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
// function scale

/******************************************************************************//**
 * \tparam LengthI    number of elements in the i-th index direction
 * \tparam LengthJ    number of elements in the j-th index direction
 * \tparam ScalarT    POD type
 * \tparam AViewTypeT view type
 * \tparam BViewTypeT view type
 *
 * \fn device_type inline void dot
 *
 * \brief Compute two-dimensional tensor dot product for each cell.
 *
 * \param [in]  aCellOrdinal cell/element ordinal
 * \param [in]  aTensorA  input 3D scalar view
 * \param [in]  aTensorB  input 3D scalar view
 * \param [out] aOutput   output 1D scalar view
**********************************************************************************/
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
// function dot

}
// namespace blas3

namespace blas1
{

/******************************************************************************//**
 * \fn device_type inline void dot
 *
 * \brief Compute absolute value of a one-dimensional scalar array
 *
 * \param [in/out] aVector 1D scalar view
**********************************************************************************/
inline void abs(const Plato::ScalarVector & aVector)
{
    Plato::OrdinalType tLength = aVector.size();
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tLength), LAMBDA_EXPRESSION(const Plato::OrdinalType & aOrdinal)
    {
        aVector(aOrdinal) = fabs(aVector(aOrdinal));
    }, "calculate absolute value");
}
// function abs

/******************************************************************************//**
 * \tparam Length    number of elements along summation dimension
 * \tparam aAlpha    multiplication/scaling factor
 * \tparam aInputWS  2D scalar view
 * \tparam aBeta     multiplication/scaling factor
 * \tparam aOutputWS 2D scalar view
 *
 * \fn device_type inline void update
 *
 * \brief Add two-dimensional 2D views, b = \alpha a + \beta b
 *
 * \param [in/out] aVector 1D scalar view
**********************************************************************************/
template<Plato::OrdinalType Length,
         typename ScalarT,
         typename AViewTypeT,
         typename BViewTypeT>
DEVICE_TYPE inline void
update
(const Plato::OrdinalType & aCellOrdinal,
 const ScalarT & aAlpha,
 const Plato::ScalarMultiVectorT<AViewTypeT> & aInputWS,
 const ScalarT & aBeta,
 const Plato::ScalarMultiVectorT<BViewTypeT> & aOutputWS)
{
    for(Plato::OrdinalType tIndex = 0; tIndex < Length; tIndex++)
    {
        aOutputWS(aCellOrdinal, tIndex) =
            aBeta * aOutputWS(aCellOrdinal, tIndex) + aAlpha * aInputWS(aCellOrdinal, tIndex);
    }
}
// update function

/******************************************************************************//**
 * \tparam Length   number of elements along summation dimension
 * \tparam ScalarT  multiplication/scaling factor
 * \tparam ResultT  2D scalar view
 *
 * \fn device_type inline void update
 *
 * \brief Scale all the elements by input scalar value, \mathbf{a} = \alpha\mathbf{a}
 *
 * \param [in]  aCellOrdinal cell/element ordinal
 * \param [in]  aScalar      acalar multiplication factor
 * \param [out] aOutputWS    output 2D scalar view
**********************************************************************************/
template<Plato::OrdinalType Length,
         typename ScalarT,
         typename ResultT>
DEVICE_TYPE inline void
scale
(const Plato::OrdinalType & aCellOrdinal,
 const ScalarT & aScalar,
 const Plato::ScalarMultiVectorT<ResultT> & aOutputWS)
{
    for(Plato::OrdinalType tIndex = 0; tIndex < Length; tIndex++)
    {
        aOutputWS(aCellOrdinal, tIndex) *= aScalar;
    }
}
// function scale

/******************************************************************************//**
 * \tparam Length   number of elements along summation dimension
 * \tparam ScalarT  multiplication/scaling factor
 * \tparam ResultT  2D scalar view
 *
 * \fn device_type inline void update
 *
 * \brief Scale all the elements by input scalar value, \mathbf{b} = \alpha\mathbf{a}
 *
 * \param [in]  aCellOrdinal cell/element ordinal
 * \param [in]  aScalar      acalar multiplication factor
 * \param [in]  aInputWS     input 2D scalar view
 * \param [out] aOutputWS    output 2D scalar view
**********************************************************************************/
template<Plato::OrdinalType Length,
         typename ScalarT,
         typename AViewTypeT,
         typename BViewTypeT>
DEVICE_TYPE inline void
scale
(const Plato::OrdinalType & aCellOrdinal,
 const ScalarT & aScalar,
 const Plato::ScalarMultiVectorT<AViewTypeT> & aInputWS,
 const Plato::ScalarMultiVectorT<BViewTypeT> & aOutputWS)
{
    for(Plato::OrdinalType tIndex = 0; tIndex < Length; tIndex++)
    {
        aOutputWS(aCellOrdinal, tIndex) = aScalar * aInputWS(aCellOrdinal, tIndex);
    }
}
// function scale

/******************************************************************************//**
 * \tparam Length     number of elements in the summing direction
 * \tparam AViewTypeT view type
 * \tparam BViewTypeT view type
 * \tparam CViewTypeT view type
 *
 * \fn device_type inline void dot
 *
 * \brief Compute two-dimensional tensor dot product for each cell.
 *
 * \param [in]  aCellOrdinal cell/element ordinal
 * \param [in]  aViewA       input 2D scalar view
 * \param [in]  aViewB       input 2D scalar view
 * \param [out] aOutput      output 1D scalar view
**********************************************************************************/
template<Plato::OrdinalType Length,
         typename AViewType,
         typename BViewType,
         typename CViewType>
DEVICE_TYPE inline void dot
(const Plato::OrdinalType & aCellOrdinal,
 const Plato::ScalarMultiVectorT<AViewType> & aViewA,
 const Plato::ScalarMultiVectorT<BViewType> & aViewB,
 const Plato::ScalarVectorT<CViewType>      & aOutput)
{
    for(Plato::OrdinalType tIndex = 0; tIndex < Length; tIndex++)
    {
        aOutput(aCellOrdinal) += aViewA(aCellOrdinal, tIndex) * aViewB(aCellOrdinal, tIndex);
    }
}
// function dot

}
// namespace blas1


/******************************************************************************//**
 * \tparam ViewType view type
 *
 * \fn inline void print_fad_val_values
 *
 * \brief Print values of 1D view of forward automatic differentiation (FAD) types.
 *
 * \param [in] aInput input 1D FAD view
 * \param [in] aName  name used to identify 1D view
**********************************************************************************/
template <typename ViewType>
inline void print_fad_val_values
(const Plato::ScalarVectorT<ViewType> & aInput,
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
// function print_fad_val_values

/******************************************************************************//**
 * \tparam NumNodesPerCell number of nodes per cell (integer)
 * \tparam NumDofsPerNode  number of degrees of freedom (integer)
 * \tparam ViewType        view type
 *
 * \fn inline void print_fad_dx_values
 *
 * \brief Print derivaitve values of 1D view of forward automatic differentiation (FAD) types.
 *
 * \param [in] aInput input 1D FAD view
 * \param [in] aName  name used to identify 1D view
**********************************************************************************/
template <Plato::OrdinalType NumNodesPerCell,
          Plato::OrdinalType NumDofsPerNode,
          typename ViewType>
inline void print_fad_dx_values
(const Plato::ScalarVectorT<ViewType> & aInput,
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
// function print_fad_dx_values

namespace omega_h
{

/******************************************************************************//**
 * \tparam ViewType view type
 *
 * \fn inline void print_fad_dx_values
 *
 * \brief Copy Kokkos view into an Omega_h LOs array.
 *
 * \param [in] aInput input 1D view
**********************************************************************************/
template<typename ViewType>
inline Omega_h::LOs copy
(const ScalarVectorT<ViewType> & aInput)
{
    auto tLength = aInput.size();
    Omega_h::Write<ViewType> tWrite(tLength);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tLength), LAMBDA_EXPRESSION(const Plato::OrdinalType & aOrdinal)
    {
        tWrite[aOrdinal] = aInput(aOrdinal);
    }, "copy");

    return (Omega_h::LOs(tWrite));
}
// function copy

/******************************************************************************//**
 * \tparam ViewType Omega_h array type
 *
 * \fn void print
 *
 * \brief Print Omega_h array to terminal.
 *
 * \param [in] aInput Omega_h array
 * \param [in] aName  name used to identify Omega_h array
**********************************************************************************/
template<typename ViewType>
void print
(const ViewType & aInput,
 const std::string & aName)
{
    std::cout << "Start Printing Array with Name '" << aName << "'\n";
    auto tLength = aInput.size();
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tLength), LAMBDA_EXPRESSION(const Plato::OrdinalType & aOrdinal)
    {
        printf("Array(%d)=%d\n",aOrdinal,aInput[aOrdinal]);
    }, "print");
    std::cout << "Finished Printing Array with Name '" << aName << "'\n";
}
// function print

/******************************************************************************//**
 * \tparam NumSpatialDims  number of spatial dimensions
 * \tparam NumNodesPerCell number of nodes per cell/element
 *
 * \fn Scalar calculate_element_size
 *
 * \brief Calculate characteristic element size
 *
 * \param [in] aCellOrdinal cell/element ordinal
 * \param [in] aCells2Nodes map from cells to node ordinal
 * \param [in] aCoords      cell/element coordinates
**********************************************************************************/
template<Plato::OrdinalType NumSpatialDims,
         Plato::OrdinalType NumNodesPerCell>
DEVICE_TYPE inline
Plato::Scalar
calculate_element_size
(const Plato::OrdinalType & aCellOrdinal,
 const Omega_h::LOs & aCells2Nodes,
 const Omega_h::Reals & aCoords)
{
    Omega_h::Few<Omega_h::Vector<NumSpatialDims>, NumNodesPerCell> tElemCoords;
    for(Plato::OrdinalType tNode = 0; tNode < NumNodesPerCell; tNode++)
    {
        const Plato::OrdinalType tVertexIndex = aCells2Nodes[aCellOrdinal*NumNodesPerCell + tNode];
        for(Plato::OrdinalType tDim = 0; tDim < NumSpatialDims; tDim++)
        {
            tElemCoords[tNode][tDim] = aCoords[tVertexIndex*NumSpatialDims + tDim];
        }
    }
    auto tSphere = Omega_h::get_inball(tElemCoords);

    return (static_cast<Plato::Scalar>(2.0) * tSphere.r);
}
// function calculate_element_size

}
// namespace omega_h

/******************************************************************************//**
 * \tparam EntitySet  entity set type (Omega_h::EntitySet)
 *
 * \fn Omega_h::LOs entity_ordinals
 *
 * \brief Return array with local entity identification numbers.
 *
 * \param [in] aMeshSets cell/element ordinal
 * \param [in] aSetName  map from cells to node ordinal
 * \param [in] aThrow    cell/element coordinates
 * \return array with local entity identification numbers
**********************************************************************************/
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
// function entity_ordinals


/******************************************************************************//**
 * \fn void is_material_defined
 *
 * \brief Check if material is defined in input file and throw an error if it is not deifned.
 *
 * \param [in] aMaterialName material sublist name
 * \param [in] aInputs       parameter list with input data information
**********************************************************************************/
inline void is_material_defined
(const std::string & aMaterialName,
 const Teuchos::ParameterList & aInputs)
{
    if(!aInputs.sublist("Material Models").isSublist(aMaterialName))
    {
        THROWERR(std::string("Material with tag '") + aMaterialName + "' is not defined in 'Material Models' block")
    }
}
// function is_material_defined


/******************************************************************************//**
 * \fn inline void is_entity_set_defined
 *
 * \brief Return true if entity set, e.g. node or side set, is defined.
 *
 * \param [in] aMeshSets list with all entity sets
 * \param [in] aSetName  entity set name
 * \return boolean
**********************************************************************************/
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
// function is_entity_set_defined


/******************************************************************************//**
 * \tparam EntitySet entity set type
 *
 * \fn inline Omega_h::LOs get_entity_ordinals
 *
 * \brief Return list of entity ordinals. If not defined, throw error to terminal.
 *
 * \param [in] aMeshSets list with all entity sets
 * \param [in] aSetName  entity set name
 * \param [in] aThrow    flag to enable throw mechanism (default = true)
 * \return list of entity ordinals for this entity set
**********************************************************************************/
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
// function get_entity_ordinals


/******************************************************************************//**
 * \fn inline Plato::OrdinalType get_num_entities
 *
 * \brief Return total number of entities in the mesh.
 *
 * \param [in] aEntityDim entity dimension (vertex, edge, face, or region)
 * \param [in] aMesh      computational mesh metadata
 * \return total number of entities in the mesh
**********************************************************************************/
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
// function get_num_entities


/******************************************************************************//**
 * \tparam EntityDim entity dimension (e.g. vertex, edge, face, or region)
 * \tparam EntitySet entity set type (e.g. nodeset or sideset)
 *
 * \fn inline Omega_h::LOs find_entities_on_non_prescribed_boundary
 *
 * \brief Return list of entity ordinals on non-prescribed boundary. A prescribed
 *   boundary is defined as the boundary where user-defined Neumann and Dirichlet
 *   boundary conditions are applied.
 *
 * \param [in] aEntitySetNames list of prescribed entity set names
 * \param [in] aMesh           computational mesh metadata
 * \param [in] aMeshSets       list of mesh sets
 *
 * \return list of entity ordinals on non-prescribed boundary
**********************************************************************************/
template
<Omega_h::Int EntityDim,
 Omega_h::SetType EntitySet>
inline Omega_h::LOs
find_entities_on_non_prescribed_boundary
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
// function find_entities_on_non_prescribed_boundary


/******************************************************************************//**
 * \tparam Type array type
 *
 * \fn inline std::vector<Type> parse_array
 *
 * \brief Return array of type=Type parsed from input file.
 *
 * \param [in] aTag    input array tag
 * \param [in] aInputs input file metadata
 *
 * \return array of type=Type
**********************************************************************************/
template <typename Type>
inline std::vector<Type>
parse_array
(const std::string & aTag,
 const Teuchos::ParameterList & aInputs)
{
    if(!aInputs.isParameter(aTag))
    {
        std::vector<Type> tOutput;
        return tOutput;
    }
    auto tSideSets = aInputs.get< Teuchos::Array<Type> >(aTag);

    auto tLength = tSideSets.size();
    std::vector<Type> tOutput(tLength);
    for(auto & tName : tOutput)
    {
        auto tIndex = &tName - &tOutput[0];
        tOutput[tIndex] = tSideSets[tIndex];
    }
    return tOutput;
}
// function parse_array


/******************************************************************************//**
 * \tparam Type parameter type
 *
 * \fn inline Type parse_parameter
 *
 * \brief Return parameter of type=Type parsed from input file.
 *
 * \param [in] aTag    input array tag
 * \param [in] aBlock  XML sublist tag
 * \param [in] aInputs input file metadata
 *
 * \return parameter of type=Type
**********************************************************************************/
template <typename Type>
inline Type parse_parameter
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
    auto tOutput = tSublist.get<Type>(aTag);
    return tOutput;
}
// function parse_parameter




/***************************************************************************//**
 *  \brief Base class for simplex-based fluid mechanics problems
 *  \tparam SpaceDim    (integer) spatial dimensions
 *  \tparam NumControls (integer) number of design variable fields (default = 1)
 ******************************************************************************/
template<Plato::OrdinalType SpaceDim, Plato::OrdinalType NumControls = 1>
class SimplexFluids: public Plato::Simplex<SpaceDim>
{
public:
    using Plato::Simplex<SpaceDim>::mNumSpatialDims;  /*!< number of spatial dimensions */
    using Plato::Simplex<SpaceDim>::mNumNodesPerCell; /*!< number of nodes per simplex cell */

    // optimization quantities of interest
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


/***************************************************************************//**
 * \struct Solutions
 *  \brief C++ structure with POD state solution data
 ******************************************************************************/
struct Solutions
{
private:
    std::unordered_map<std::string, Plato::ScalarMultiVector> mSolution; /*!< map from state solution name to 2D POD array */

public:
    /***************************************************************************//**
     * \fn Plato::OrdinalType size
     *
     * \brief Return number of elements in solution map.
     * \return number of elements in solution map (integer)
     ******************************************************************************/
    Plato::OrdinalType size() const
    {
        return (mSolution.size());
    }

    /***************************************************************************//**
     * \fn std::vector<std::string> tags
     *
     * \brief Return list with state solution tags.
     * \return list with state solution tags
     ******************************************************************************/
    std::vector<std::string> tags() const
    {
        std::vector<std::string> tTags;
        for(auto& tPair : mSolution)
        {
            tTags.push_back(tPair.first);
        }
        return tTags;
    }

    /***************************************************************************//**
     * \fn void set
     *
     * \brief Set value of an element in the solution map.
     * \param aTag  data tag
     * \param aData 2D POD array
     ******************************************************************************/
    void set(const std::string& aTag, const Plato::ScalarMultiVector& aData)
    {
        auto tLowerTag = Plato::tolower(aTag);
        mSolution[tLowerTag] = aData;
    }

    /***************************************************************************//**
     * \fn Plato::ScalarMultiVector get
     *
     * \brief Return 2D POD array.
     * \param aTag data tag
     ******************************************************************************/
    Plato::ScalarMultiVector get(const std::string& aTag) const
    {
        auto tLowerTag = Plato::tolower(aTag);
        auto tItr = mSolution.find(tLowerTag);
        if(tItr == mSolution.end())
        {
            THROWERR(std::string("Solution with tag '") + aTag + "' is not defined.")
        }
        return tItr->second;
    }
};
// struct Solutions


/***************************************************************************//**
 *  \class MetaDataBase
 *  \brief Plato metadata pure virtual base class.
 ******************************************************************************/
class MetaDataBase
{
public:
    virtual ~MetaDataBase() = 0;
};
inline MetaDataBase::~MetaDataBase(){}
// class MetaDataBase


/***************************************************************************//**
 * \tparam Type metadata type
 * \class MetaData
 * \brief Plato metadata derived class.
 ******************************************************************************/
template<class Type>
class MetaData : public MetaDataBase
{
public:
    /***************************************************************************//**
     * \brief Constructor
     * \param aData metadata
     ******************************************************************************/
    explicit MetaData(const Type &aData) : mData(aData) {}
    MetaData() {}
    Type mData; /*!< metadata */
};
// class MetaData


/***************************************************************************//**
 * \tparam Type metadata type
 *
 * \fn inline Type metadata
 *
 * \brief Perform dynamic cast from MetaDataBase to Type data.
 *
 * \param aInput shared pointer of Plato metadata
 * \return Type data
 ******************************************************************************/
template<class Type>
inline Type metadata(const std::shared_ptr<Plato::MetaDataBase> & aInput)
{
    return (dynamic_cast<Plato::MetaData<Type>&>(aInput.operator*()).mData);
}
// function metadata


/***************************************************************************//**
 * \struct WorkSets
 * \brief Map with Plato metadata worksets.
 ******************************************************************************/
struct WorkSets
{
private:
    std::unordered_map<std::string, std::shared_ptr<Plato::MetaDataBase>> mData; /*!< map from tag to metadata shared pointer */

public:
    WorkSets(){}

    /***************************************************************************//**
     * \fn void set
     * \brief Set element metadata at input key location.
     * \param aName metadata tag (i.e. key)
     * \param aData metadata shared pointer
     ******************************************************************************/
    void set(const std::string & aName, const std::shared_ptr<Plato::MetaDataBase> & aData)
    {
        auto tLowerKey = Plato::tolower(aName);
        mData[tLowerKey] = aData;
    }

    /***************************************************************************//**
     * \fn const std::shared_ptr<Plato::MetaDataBase> & get
     * \brief Return const reference to metadata shared pointer at input key location.
     * \param aName metadata tag (i.e. key)
     * \return metadata shared pointer
     ******************************************************************************/
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

    /***************************************************************************//**
     * \fn std::vector<std::string> tags
     * \brief Return list of keys/tags in the metadata map.
     * \return list of keys/tags in the metadata map
     ******************************************************************************/
    std::vector<std::string> tags() const
    {
        std::vector<std::string> tOutput;
        for(auto& tPair : mData)
        {
            tOutput.push_back(tPair.first);
        }
        return tOutput;
    }

    /***************************************************************************//**
     * \fn bool defined
     * \brief Return true is key/metadata pair is defined in the metadata map, else
     *   return false.
     * \return boolean (true or false)
     ******************************************************************************/
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
// struct WorkSets



/***************************************************************************//**
 * \tparam PhysicsT physics type, e.g. fluid, mechancis, thermal, etc.
 *
 * \struct LocalOrdinalMaps
 *
 * \brief Collection of ordinal id maps for scalar, vector, and control fields.
 ******************************************************************************/
template <typename PhysicsT>
struct LocalOrdinalMaps
{
    Plato::NodeCoordinate<PhysicsT::SimplexT::mNumSpatialDims> mNodeCoordinate; /*!< list of node coordinates */
    Plato::VectorEntryOrdinal<PhysicsT::SimplexT::mNumSpatialDims, 1 /*scalar dofs per node*/>                 mScalarFieldOrdinalsMap; /*!< element to scalar field degree of freedom map */
    Plato::VectorEntryOrdinal<PhysicsT::SimplexT::mNumSpatialDims, PhysicsT::SimplexT::mNumSpatialDims>        mVectorFieldOrdinalsMap; /*!< element to vector field degree of freedom map */
    Plato::VectorEntryOrdinal<PhysicsT::SimplexT::mNumSpatialDims, PhysicsT::SimplexT::mNumControlDofsPerNode> mControlOrdinalsMap; /*!< element to control field degree of freedom map */

    /***************************************************************************//**
     * \fn LocalOrdinalMaps
     *
     * \brief Constructor
     * \param [in] aMesh mesh metadata
     ******************************************************************************/
    LocalOrdinalMaps(Omega_h::Mesh & aMesh) :
        mNodeCoordinate(&aMesh),
        mScalarFieldOrdinalsMap(&aMesh),
        mVectorFieldOrdinalsMap(&aMesh),
        mControlOrdinalsMap(&aMesh)
    { return; }
};
// struct LocalOrdinalMaps


/***************************************************************************//**
 * \struct Variables
 *
 * \brief Maps to quantity of interest associated with the simulation.
 ******************************************************************************/
struct Variables
{
private:
    std::unordered_map<std::string, Plato::Scalar> mScalars; /*!< map to scalar quantities of interest */
    std::unordered_map<std::string, Plato::ScalarVector> mVectors; /*!< map to vector quantities of interest */

public:
    /***************************************************************************//**
     * \fn void scalar
     * \brief Return scalar value associated with this tag.
     * \param [in] aTag element tag/key
     ******************************************************************************/
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

    /***************************************************************************//**
     * \fn void scalar
     * \brief Set (element,key) pair in scalar value map.
     * \param [in] aTag   element tag/key
     * \param [in] aInput element value
     ******************************************************************************/
    void scalar(const std::string& aTag, const Plato::Scalar& aInput)
    {
        auto tLowerTag = Plato::tolower(aTag);
        mScalars[tLowerTag] = aInput;
    }

    /***************************************************************************//**
     * \fn void vector
     * \brief Return scalar vector associated with this tag/key.
     * \param [in] aTag element tag/key
     ******************************************************************************/
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

    /***************************************************************************//**
     * \fn void vector
     * \brief Set (element,key) pair in vector value map.
     * \param [in] aTag   element tag/key
     * \param [in] aInput element value
     ******************************************************************************/
    void vector(const std::string& aTag, const Plato::ScalarVector& aInput)
    {
        auto tLowerTag = Plato::tolower(aTag);
        mVectors[tLowerTag] = aInput;
    }

    /***************************************************************************//**
     * \fn bool isVectorMapEmpty
     * \brief Returns true if vector map is empty; false, if not empty.
     * \return boolean (true or false)
     ******************************************************************************/
    bool isVectorMapEmpty() const
    {
        return mVectors.empty();
    }

    /***************************************************************************//**
     * \fn bool isScalarMapEmpty
     * \brief Returns true if scalar map is empty; false, if not empty.
     * \return boolean (true or false)
     ******************************************************************************/
    bool isScalarMapEmpty() const
    {
        return mScalars.empty();
    }

    /***************************************************************************//**
     * \fn bool defined
     * \brief Returns true if element with tak/key is defined in a map.
     * \return boolean (true or false)
     ******************************************************************************/
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
// struct Variables
typedef Variables Dual;   /*!< variant name used for the Variables structure to identify quantities associated with the dual problem in optimization */
typedef Variables Primal; /*!< variant name used for the Variables structure to identify quantities associated with the primal problem in optimization */


namespace Fluids
{

/***************************************************************************//**
 * \tparam SimplexPhysics physics type associated with simplex elements
 *
 * \struct SimplexFadTypes
 *
 * \brief The C++ structure owns the Forward Automatic Differentiation (FAD)
 * types used for the Quantities of Interest (QoI) in fluid flow applications.
 ******************************************************************************/
template<typename SimplexPhysics>
struct SimplexFadTypes
{
    using ConfigFad   = Sacado::Fad::SFad<Plato::Scalar, SimplexPhysics::mNumConfigDofsPerCell>;   /*!< configuration FAD type */
    using ControlFad  = Sacado::Fad::SFad<Plato::Scalar, SimplexPhysics::mNumNodesPerCell>;        /*!< control FAD type */
    using MassFad     = Sacado::Fad::SFad<Plato::Scalar, SimplexPhysics::mNumMassDofsPerCell>;     /*!< mass QoI FAD type */
    using EnergyFad   = Sacado::Fad::SFad<Plato::Scalar, SimplexPhysics::mNumEnergyDofsPerCell>;   /*!< energy QoI FAD type */
    using MomentumFad = Sacado::Fad::SFad<Plato::Scalar, SimplexPhysics::mNumMomentumDofsPerCell>; /*!< momentum QoI FAD type */
};
// struct SimplexFadTypes

/***************************************************************************//**
 * \tparam SimplexFadTypesT physics type associated with simplex elements
 * \tparam ScalarType       scalar type
 *
 * \struct is_fad<SimplexFadTypesT, ScalarType>::value
 *
 * \brief is true if ScalarType is of any AD type defined in SimplexFadTypesT.
 ******************************************************************************/
template <typename SimplexFadTypesT, typename ScalarType>
struct is_fad {
  static constexpr bool value = std::is_same< ScalarType, typename SimplexFadTypesT::MassFad     >::value ||
                                std::is_same< ScalarType, typename SimplexFadTypesT::ControlFad  >::value ||
                                std::is_same< ScalarType, typename SimplexFadTypesT::ConfigFad   >::value ||
                                std::is_same< ScalarType, typename SimplexFadTypesT::EnergyFad   >::value ||
                                std::is_same< ScalarType, typename SimplexFadTypesT::MomentumFad >::value;
};
// struct is_fad


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

/***************************************************************************//**
 * \tparam SimplexPhysicsT physics type
 *
 * \struct ResultTypes
 *
 * \brief Scalar types for residual evaluations.
 ******************************************************************************/
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
// struct ResultTypes

/***************************************************************************//**
 * \tparam SimplexPhysicsT physics type
 *
 * \struct GradCurrentMomentumTypes
 *
 * \brief Scalar types for evaluations associated with the partial derivative
 * of a vector/scalar value function with respect to the current momentum field.
 ******************************************************************************/
template <typename SimplexPhysicsT>
struct GradCurrentMomentumTypes : EvaluationTypes<SimplexPhysicsT>
{
    using FadType = typename Plato::Fluids::SimplexFadTypes<SimplexPhysicsT>::MomentumFad;

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
// struct GradCurrentMomentumTypes


/***************************************************************************//**
 * \tparam SimplexPhysicsT physics type
 *
 * \struct GradCurrentEnergyTypes
 *
 * \brief Scalar types for evaluations associated with the partial derivative
 * of a vector/scalar value function with respect to the current energy field.
 ******************************************************************************/
template <typename SimplexPhysicsT>
struct GradCurrentEnergyTypes : EvaluationTypes<SimplexPhysicsT>
{
    using FadType = typename Plato::Fluids::SimplexFadTypes<SimplexPhysicsT>::EnergyFad;

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
// struct GradCurrentEnergyTypes

/***************************************************************************//**
 * \tparam SimplexPhysicsT physics type
 *
 * \struct GradCurrentMassTypes
 *
 * \brief Scalar types for evaluations associated with the partial derivative
 * of a vector/scalar value function with respect to the current mass field.
 ******************************************************************************/
template <typename SimplexPhysicsT>
struct GradCurrentMassTypes : EvaluationTypes<SimplexPhysicsT>
{
    using FadType = typename Plato::Fluids::SimplexFadTypes<SimplexPhysicsT>::MassFad;

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
// struct GradCurrentMassTypes

/***************************************************************************//**
 * \tparam SimplexPhysicsT physics type
 *
 * \struct GradPreviousMomentumTypes
 *
 * \brief Scalar types for evaluations associated with the partial derivative
 * of a vector/scalar value function with respect to the previous momentum field.
 ******************************************************************************/
template <typename SimplexPhysicsT>
struct GradPreviousMomentumTypes : EvaluationTypes<SimplexPhysicsT>
{
    using FadType = typename Plato::Fluids::SimplexFadTypes<SimplexPhysicsT>::MomentumFad;

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
// struct GradPreviousMomentumTypes


/***************************************************************************//**
 * \tparam SimplexPhysicsT physics type
 *
 * \struct GradPreviousEnergyTypes
 *
 * \brief Scalar types for evaluations associated with the partial derivative
 * of a vector/scalar value function with respect to the previous energy field.
 ******************************************************************************/
template <typename SimplexPhysicsT>
struct GradPreviousEnergyTypes : EvaluationTypes<SimplexPhysicsT>
{
    using FadType = typename Plato::Fluids::SimplexFadTypes<SimplexPhysicsT>::EnergyFad;

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
// struct GradPreviousEnergyTypes

/***************************************************************************//**
 * \tparam SimplexPhysicsT physics type
 *
 * \struct GradPreviousMassTypes
 *
 * \brief Scalar types for evaluations associated with the partial derivative
 * of a vector/scalar value function with respect to the previous mass field.
 ******************************************************************************/
template <typename SimplexPhysicsT>
struct GradPreviousMassTypes : EvaluationTypes<SimplexPhysicsT>
{
    using FadType = typename Plato::Fluids::SimplexFadTypes<SimplexPhysicsT>::MassFad;

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
// struct GradPreviousMassTypes

/***************************************************************************//**
 * \tparam SimplexPhysicsT physics type
 *
 * \struct GradMomentumPredictorTypes
 *
 * \brief Scalar types for evaluations associated with the partial derivative
 * of a vector/scalar value function with respect to the predictor field.
 ******************************************************************************/
template <typename SimplexPhysicsT>
struct GradMomentumPredictorTypes : EvaluationTypes<SimplexPhysicsT>
{
    using FadType = typename Plato::Fluids::SimplexFadTypes<SimplexPhysicsT>::MomentumFad;

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
// struct GradMomentumPredictorTypes

/***************************************************************************//**
 * \tparam SimplexPhysicsT physics type
 *
 * \struct GradConfigTypes
 *
 * \brief Scalar types for evaluations associated with the partial derivative
 * of a vector/scalar value function with respect to configuration variables.
 ******************************************************************************/
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
// struct GradConfigTypes

/***************************************************************************//**
 * \tparam SimplexPhysicsT physics type
 *
 * \struct GradControlTypes
 *
 * \brief Scalar types for evaluations associated with the partial derivative
 * of a vector/scalar value function with respect to control variables.
 ******************************************************************************/
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
// struct GradControlTypes


/***************************************************************************//**
 * \tparam SimplexPhysicsT physics type
 *
 * \struct Evaluation
 *
 * \brief Wrapper structure for the evaluation types used in fluid flow applications.
 ******************************************************************************/
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
// struct Evaluation


/***************************************************************************//**
 * \tparam PhysicsT physics type
 *
 * \struct Evaluation
 *
 * \brief Functionalities in this structure are used to build data work sets.
 * The data types are assigned based on the physics and automatic differentiation
 * (AD) evaluation types used for fluid flow applications.
 ******************************************************************************/
template<typename PhysicsT>
struct WorkSetBuilder
{
private:
    using SimplexPhysicsT = typename PhysicsT::SimplexT; /*!< holds static values used in fluid flow applications solved with simplex elements */

    using ConfigLocalOridnalMap   = Plato::NodeCoordinate<SimplexPhysicsT::mNumSpatialDims>; /*!< short name used for wrapper class holding coordinate information */

    using MassLocalOridnalMap     = Plato::VectorEntryOrdinal<SimplexPhysicsT::mNumSpatialDims, SimplexPhysicsT::mNumMassDofsPerNode>; /*!< short name used for wrapper class mapping elements to local mass degrees of freedom  */
    using EnergyLocalOridnalMap   = Plato::VectorEntryOrdinal<SimplexPhysicsT::mNumSpatialDims, SimplexPhysicsT::mNumEnergyDofsPerNode>; /*!< short name used for wrapper class mapping elements to local energy degrees of freedom  */
    using MomentumLocalOridnalMap = Plato::VectorEntryOrdinal<SimplexPhysicsT::mNumSpatialDims, SimplexPhysicsT::mNumMomentumDofsPerNode>; /*!< short name used for wrapper class mapping elements to local momentum degrees of freedom  */
    using ControlLocalOridnalMap  = Plato::VectorEntryOrdinal<SimplexPhysicsT::mNumSpatialDims, SimplexPhysicsT::mNumControlDofsPerNode>; /*!< short name used for wrapper class mapping elements to local control degrees of freedom  */

    using ConfigFad   = typename Plato::Fluids::SimplexFadTypes<SimplexPhysicsT>::ConfigFad; /*!< configuration forward AD type  */
    using ControlFad  = typename Plato::Fluids::SimplexFadTypes<SimplexPhysicsT>::ControlFad; /*!< control forward AD type  */
    using MassFad     = typename Plato::Fluids::SimplexFadTypes<SimplexPhysicsT>::MassFad; /*!< mass forward AD type  */
    using EnergyFad   = typename Plato::Fluids::SimplexFadTypes<SimplexPhysicsT>::EnergyFad; /*!< energy forward AD type  */
    using MomentumFad = typename Plato::Fluids::SimplexFadTypes<SimplexPhysicsT>::MomentumFad; /*!< momentum forward AD type  */

public:
    /***************************************************************************//**
     * \fn void buildMomentumWorkSet
     *
     * \brief build momentum field work set of POD type.
     *
     * \param [in] aDomain structure with computational domain metadata such as the mesh and entity sets.
     * \param [in] aMap    map from element to local momentum degrees of freedom
     * \param [in] aInput  one dimensional momentum field, i.e. flatten momentum field
     *
     * \param [in/out] aOutput two dimensional momentum work set
     ******************************************************************************/
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

    /***************************************************************************//**
     * \fn void buildMomentumWorkSet
     *
     * \brief build momentum field work set of POD type.
     *
     * \param [in] aNumCells number of cells/elements
     * \param [in] aMap      map from element to local momentum degrees of freedom
     * \param [in] aInput    one dimensional momentum field, i.e. flatten momentum field
     *
     * \param [in/out] aOutput two dimensional momentum work set
     ******************************************************************************/
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

    /***************************************************************************//**
     * \fn void buildMomentumWorkSet
     *
     * \brief build momentum field work set of FAD type.
     *
     * \param [in] aDomain structure with computational domain metadata such as the mesh and entity sets.
     * \param [in] aMap    map from element to local momentum degrees of freedom
     * \param [in] aInput  one dimensional momentum field, i.e. flatten momentum field
     *
     * \param [in/out] aOutput two dimensional momentum work set
     ******************************************************************************/
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

    /***************************************************************************//**
     * \fn void buildMomentumWorkSet
     *
     * \brief build momentum field work set of FAD type.
     *
     * \param [in] aNumCells number of cells/elements
     * \param [in] aMap      map from element to local momentum degrees of freedom
     * \param [in] aInput    one dimensional momentum field, i.e. flatten momentum field
     *
     * \param [in/out] aOutput two dimensional momentum work set
     ******************************************************************************/
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

    /***************************************************************************//**
     * \fn void buildEnergyWorkSet
     *
     * \brief build energy field work set of POD type.
     *
     * \param [in] aDomain structure with computational domain metadata such as the mesh and entity sets.
     * \param [in] aMap    map from element to local energy degrees of freedom
     * \param [in] aInput  one dimensional energy field, i.e. flatten energy field
     *
     * \param [in/out] aOutput two dimensional energy work set
     ******************************************************************************/
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

    /***************************************************************************//**
     * \fn void buildEnergyWorkSet
     *
     * \brief build energy field work set of POD type.
     *
     * \param [in] aNumCells number of cells/elements
     * \param [in] aMap      map from element to local energy degrees of freedom
     * \param [in] aInput    one dimensional energy field, i.e. flatten energy field
     *
     * \param [in/out] aOutput two dimensional energy work set
     ******************************************************************************/
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

    /***************************************************************************//**
     * \fn void buildEnergyWorkSet
     *
     * \brief build energy field work set of FAD type.
     *
     * \param [in] aDomain structure with computational domain metadata such as the mesh and entity sets.
     * \param [in] aMap    map from element to local energy degrees of freedom
     * \param [in] aInput  one dimensional energy field, i.e. flatten energy field
     *
     * \param [in/out] aOutput two dimensional energy work set
     ******************************************************************************/
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

    /***************************************************************************//**
     * \fn void buildEnergyWorkSet
     *
     * \brief build energy field work set of FAD type.
     *
     * \param [in] aNumCells number of cells/elements
     * \param [in] aMap      map from element to local energy degrees of freedom
     * \param [in] aInput    one dimensional energy field, i.e. flatten energy field
     *
     * \param [in/out] aOutput two dimensional energy work set
     ******************************************************************************/
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

    /***************************************************************************//**
     * \fn void buildMassWorkSet
     *
     * \brief build mass field work set of POD type.
     *
     * \param [in] aDomain structure with computational domain metadata such as the mesh and entity sets.
     * \param [in] aMap    map from element to local mass degrees of freedom
     * \param [in] aInput  one dimensional mass field, i.e. flatten mass field
     *
     * \param [in/out] aOutput two dimensional mass work set
     ******************************************************************************/
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

    /***************************************************************************//**
     * \fn void buildMassWorkSet
     *
     * \brief build mass field work set of POD type.
     *
     * \param [in] aNumCells number of cells/elements
     * \param [in] aMap      map from element to local mass degrees of freedom
     * \param [in] aInput    one dimensional mass field, i.e. flatten mass field
     *
     * \param [in/out] aOutput two dimensional mass work set
     ******************************************************************************/
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

    /***************************************************************************//**
     * \fn void buildMassWorkSet
     *
     * \brief build mass field work set of FAD type.
     *
     * \param [in] aDomain structure with computational domain metadata such as the mesh and entity sets.
     * \param [in] aMap    map from element to local mass degrees of freedom
     * \param [in] aInput  one dimensional mass field, i.e. flatten mass field
     *
     * \param [in/out] aOutput two dimensional mass work set
     ******************************************************************************/
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

    /***************************************************************************//**
     * \fn void buildMassWorkSet
     *
     * \brief build mass field work set of FAD type.
     *
     * \param [in] aNumCells number of cells/elements
     * \param [in] aMap      map from element to local mass degrees of freedom
     * \param [in] aInput    one dimensional mass field, i.e. flatten mass field
     *
     * \param [in/out] aOutput two dimensional mass work set
     ******************************************************************************/
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

    /***************************************************************************//**
     * \fn void buildControlWorkSet
     *
     * \brief build control field work set of POD type.
     *
     * \param [in] aDomain structure with computational domain metadata such as the mesh and entity sets.
     * \param [in] aMap    map from element to local control degrees of freedom
     * \param [in] aInput  one dimensional control field, i.e. flatten control field
     *
     * \param [in/out] aOutput two dimensional control work set
     ******************************************************************************/
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

    /***************************************************************************//**
     * \fn void buildControlWorkSet
     *
     * \brief build control field work set of POD type.
     *
     * \param [in] aNumCells number of cells/elements
     * \param [in] aMap      map from element to local control degrees of freedom
     * \param [in] aInput    one dimensional control field, i.e. flatten control field
     *
     * \param [in/out] aOutput two dimensional control work set
     ******************************************************************************/
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

    /***************************************************************************//**
     * \fn void buildControlWorkSet
     *
     * \brief build control field work set of FAD type.
     *
     * \param [in] aDomain structure with computational domain metadata such as the mesh and entity sets.
     * \param [in] aMap    map from element to local control degrees of freedom
     * \param [in] aInput  one dimensional control field, i.e. flatten control field
     *
     * \param [in/out] aOutput two dimensional control work set
     ******************************************************************************/
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

    /***************************************************************************//**
     * \fn void buildControlWorkSet
     *
     * \brief build control field work set of FAD type.
     *
     * \param [in] aNumCells number of cells/elements
     * \param [in] aMap      map from element to local control degrees of freedom
     * \param [in] aInput    one dimensional control field, i.e. flatten control field
     *
     * \param [in/out] aOutput two dimensional control work set
     ******************************************************************************/
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

    /***************************************************************************//**
     * \fn void buildConfigWorkSet
     *
     * \brief build configuration field work set of POD type.
     *
     * \param [in] aDomain structure with computational domain metadata such as the mesh and entity sets.
     * \param [in] aMap    map from element to local configuration degrees of freedom
     *
     * \param [in/out] aOutput three dimensional configuration work set
     ******************************************************************************/
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

    /***************************************************************************//**
     * \fn void buildConfigWorkSet
     *
     * \brief build configuration field work set of POD type.
     *
     * \param [in] aNumCells number of cells/elements
     * \param [in] aMap      map from element to local configuration degrees of freedom
     *
     * \param [in/out] aOutput three dimensional configuration work set
     ******************************************************************************/
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

    /***************************************************************************//**
     * \fn void buildConfigWorkSet
     *
     * \brief build configuration field work set of FAD type.
     *
     * \param [in] aDomain structure with computational domain metadata such as the mesh and entity sets.
     * \param [in] aMap    map from element to local configuration degrees of freedom
     *
     * \param [in/out] aOutput three dimensional configuration work set
     ******************************************************************************/
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

    /***************************************************************************//**
     * \fn void buildConfigWorkSet
     *
     * \brief build configuration field work set of FAD type.
     *
     * \param [in] aNumCells number of cells/elements
     * \param [in] aMap      map from element to local configuration degrees of freedom
     *
     * \param [in/out] aOutput three dimensional configuration work set
     ******************************************************************************/
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
// struct WorkSetBuilder



/***************************************************************************//**
 * \tparam EvaluationT Forward Automatic Differentiation (FAD) evaluation type
 * \tparam PhysicsT    fluid flow physics type
 *
 * \fn inline void build_scalar_function_worksets
 *
 * \brief Build metadata work sets used for the evaluation of a scalar function for fluid flow applications.
 *
 * \param [in] aDomain    computational domain metadata ( e.g. mesh and entity sets)
 * \param [in] aControls  1D array of control field
 * \param [in] aVariables state metadata (e.g. pressure, velocity, temperature, etc.)
 * \param [in] aMaps      holds maps from element to local field degrees of freedom
 *
 * \param [in/out] aWorkSets state work sets initialize with the correct FAD type
 ******************************************************************************/
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
    Plato::Fluids::WorkSetBuilder<PhysicsT> tWorkSetBuilder;

    using CurrentVelocityT = typename EvaluationT::CurrentMomentumScalarType;
    auto tCurVelWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<CurrentVelocityT> > >
        ( Plato::ScalarMultiVectorT<CurrentVelocityT>("current velocity", tNumCells, PhysicsT::SimplexT::mNumMomentumDofsPerCell) );
    tWorkSetBuilder.buildMomentumWorkSet(aDomain, aMaps.mVectorFieldOrdinalsMap, aVariables.vector("current velocity"), tCurVelWS->mData);
    aWorkSets.set("current velocity", tCurVelWS);

    using CurrentPressureT = typename EvaluationT::CurrentMassScalarType;
    auto tCurPressWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<CurrentPressureT> > >
        ( Plato::ScalarMultiVectorT<CurrentPressureT>("current pressure", tNumCells, PhysicsT::SimplexT::mNumMassDofsPerCell) );
    tWorkSetBuilder.buildMassWorkSet(aDomain, aMaps.mScalarFieldOrdinalsMap, aVariables.vector("current pressure"), tCurPressWS->mData);
    aWorkSets.set("current pressure", tCurPressWS);

    using CurrentTemperatureT = typename EvaluationT::CurrentEnergyScalarType;
    auto tCurTempWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<CurrentTemperatureT> > >
        ( Plato::ScalarMultiVectorT<CurrentTemperatureT>("current temperature", tNumCells, PhysicsT::SimplexT::mNumEnergyDofsPerCell) );
    tWorkSetBuilder.buildEnergyWorkSet(aDomain, aMaps.mScalarFieldOrdinalsMap, aVariables.vector("current temperature"), tCurTempWS->mData);
    aWorkSets.set("current temperature", tCurTempWS);

    using ControlT = typename EvaluationT::ControlScalarType;
    auto tControlWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<ControlT> > >
        ( Plato::ScalarMultiVectorT<ControlT>("control", tNumCells, PhysicsT::SimplexT::mNumNodesPerCell) );
    tWorkSetBuilder.buildControlWorkSet(aDomain, aMaps.mControlOrdinalsMap, aControls, tControlWS->mData);
    aWorkSets.set("control", tControlWS);

    auto tCriticalTimeStep = std::make_shared< Plato::MetaData< Plato::ScalarVector > >( Plato::ScalarVector("critical time step", 1) );
    Plato::blas1::copy(aVariables.vector("critical time step"), tCriticalTimeStep->mData);
    aWorkSets.set("critical time step", tCriticalTimeStep);

    using ConfigT = typename EvaluationT::ConfigScalarType;
    auto tConfig = std::make_shared< Plato::MetaData< Plato::ScalarArray3DT<ConfigT> > >
        ( Plato::ScalarArray3DT<ConfigT>("configuration", tNumCells, PhysicsT::SimplexT::mNumNodesPerCell, PhysicsT::SimplexT::mNumConfigDofsPerNode) );
    tWorkSetBuilder.buildConfigWorkSet(aDomain, aMaps.mNodeCoordinate, tConfig->mData);
    aWorkSets.set("configuration", tConfig);
}
// function build_scalar_function_worksets


/***************************************************************************//**
 * \tparam EvaluationT Forward Automatic Differentiation (FAD) evaluation type
 * \tparam PhysicsT    fluid flow physics type
 *
 * \fn inline void build_scalar_function_worksets
 *
 * \brief Build metadata work sets used for the evaluation of a scalar function for fluid flow applications.
 *
 * \param [in] aNumCells  total number of cells
 * \param [in] aControls  1D array of control field
 * \param [in] aVariables state metadata (e.g. pressure, velocity, temperature, etc.)
 * \param [in] aMaps      holds maps from element to local field degrees of freedom
 *
 * \param [in/out] aWorkSets state work sets initialize with the correct FAD type
 ******************************************************************************/
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
    Plato::Fluids::WorkSetBuilder<PhysicsT> tWorkSetBuilder;

    using CurrentVelocityT = typename EvaluationT::CurrentMomentumScalarType;
    auto tCurVelWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<CurrentVelocityT> > >
        ( Plato::ScalarMultiVectorT<CurrentVelocityT>("current velocity", aNumCells, PhysicsT::SimplexT::mNumMomentumDofsPerCell) );
    tWorkSetBuilder.buildMomentumWorkSet(aNumCells, aMaps.mVectorFieldOrdinalsMap, aVariables.vector("current velocity"), tCurVelWS->mData);
    aWorkSets.set("current velocity", tCurVelWS);

    using CurrentPressureT = typename EvaluationT::CurrentMassScalarType;
    auto tCurPressWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<CurrentPressureT> > >
        ( Plato::ScalarMultiVectorT<CurrentPressureT>("current pressure", aNumCells, PhysicsT::SimplexT::mNumMassDofsPerCell) );
    tWorkSetBuilder.buildMassWorkSet(aNumCells, aMaps.mScalarFieldOrdinalsMap, aVariables.vector("current pressure"), tCurPressWS->mData);
    aWorkSets.set("current pressure", tCurPressWS);

    using CurrentTemperatureT = typename EvaluationT::CurrentEnergyScalarType;
    auto tCurTempWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<CurrentTemperatureT> > >
        ( Plato::ScalarMultiVectorT<CurrentTemperatureT>("current temperature", aNumCells, PhysicsT::SimplexT::mNumEnergyDofsPerCell) );
    tWorkSetBuilder.buildEnergyWorkSet(aNumCells, aMaps.mScalarFieldOrdinalsMap, aVariables.vector("current temperature"), tCurTempWS->mData);
    aWorkSets.set("current temperature", tCurTempWS);

    using ControlT = typename EvaluationT::ControlScalarType;
    auto tControlWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<ControlT> > >
        ( Plato::ScalarMultiVectorT<ControlT>("control", aNumCells, PhysicsT::SimplexT::mNumNodesPerCell) );
    tWorkSetBuilder.buildControlWorkSet(aNumCells, aMaps.mControlOrdinalsMap, aControls, tControlWS->mData);
    aWorkSets.set("control", tControlWS);

    auto tCriticalTimeStep = std::make_shared< Plato::MetaData< Plato::ScalarVector > >( Plato::ScalarVector("critical time step", 1) );
    Plato::blas1::copy(aVariables.vector("critical time step"), tCriticalTimeStep->mData);
    aWorkSets.set("critical time step", tCriticalTimeStep);

    using ConfigT = typename EvaluationT::ConfigScalarType;
    auto tConfig = std::make_shared< Plato::MetaData< Plato::ScalarArray3DT<ConfigT> > >
        ( Plato::ScalarArray3DT<ConfigT>("configuration", aNumCells, PhysicsT::SimplexT::mNumNodesPerCell, PhysicsT::SimplexT::mNumConfigDofsPerNode) );
    tWorkSetBuilder.buildConfigWorkSet(aNumCells, aMaps.mNodeCoordinate, tConfig->mData);
    aWorkSets.set("configuration", tConfig);
}
// function build_scalar_function_worksets


/***************************************************************************//**
 * \tparam EvaluationT Forward Automatic Differentiation (FAD) evaluation type
 * \tparam PhysicsT    fluid flow physics type
 *
 * \fn inline void build_vector_function_worksets
 *
 * \brief Build metadata work sets used for the evaluation of a vector function for fluid flow applications.
 *
 * \param [in] aDomain    computational domain metadata ( e.g. mesh and entity sets)
 * \param [in] aControls  1D array of control field
 * \param [in] aVariables state metadata (e.g. pressure, velocity, temperature, etc.)
 * \param [in] aMaps      holds maps from element to local field degrees of freedom
 *
 * \param [in/out] aWorkSets state work sets initialize with the correct FAD type
 ******************************************************************************/
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
    Plato::Fluids::WorkSetBuilder<PhysicsT> tWorkSetBuilder;

    using CurrentPredictorT = typename EvaluationT::MomentumPredictorScalarType;
    auto tPredictorWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<CurrentPredictorT> > >
        ( Plato::ScalarMultiVectorT<CurrentPredictorT>("current predictor", tNumCells, PhysicsT::SimplexT::mNumMomentumDofsPerCell) );
    tWorkSetBuilder.buildMomentumWorkSet(aDomain, aMaps.mVectorFieldOrdinalsMap, aVariables.vector("current predictor"), tPredictorWS->mData);
    aWorkSets.set("current predictor", tPredictorWS);

    using CurrentVelocityT = typename EvaluationT::CurrentMomentumScalarType;
    auto tCurVelWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<CurrentVelocityT> > >
        ( Plato::ScalarMultiVectorT<CurrentVelocityT>("current velocity", tNumCells, PhysicsT::SimplexT::mNumMomentumDofsPerCell) );
    tWorkSetBuilder.buildMomentumWorkSet(aDomain, aMaps.mVectorFieldOrdinalsMap, aVariables.vector("current velocity"), tCurVelWS->mData);
    aWorkSets.set("current velocity", tCurVelWS);

    using CurrentPressureT = typename EvaluationT::CurrentMassScalarType;
    auto tCurPressWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<CurrentPressureT> > >
        ( Plato::ScalarMultiVectorT<CurrentPressureT>("current pressure", tNumCells, PhysicsT::SimplexT::mNumMassDofsPerCell) );
    tWorkSetBuilder.buildMassWorkSet(aDomain, aMaps.mScalarFieldOrdinalsMap, aVariables.vector("current pressure"), tCurPressWS->mData);
    aWorkSets.set("current pressure", tCurPressWS);

    using CurrentTemperatureT = typename EvaluationT::CurrentEnergyScalarType;
    auto tCurTempWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<CurrentTemperatureT> > >
        ( Plato::ScalarMultiVectorT<CurrentTemperatureT>("current temperature", tNumCells, PhysicsT::SimplexT::mNumEnergyDofsPerCell) );
    tWorkSetBuilder.buildEnergyWorkSet(aDomain, aMaps.mScalarFieldOrdinalsMap, aVariables.vector("current temperature"), tCurTempWS->mData);
    aWorkSets.set("current temperature", tCurTempWS);

    using PreviousVelocityT = typename EvaluationT::PreviousMomentumScalarType;
    auto tPrevVelWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<PreviousVelocityT> > >
        ( Plato::ScalarMultiVectorT<PreviousVelocityT>("previous velocity", tNumCells, PhysicsT::SimplexT::mNumMomentumDofsPerCell) );
    tWorkSetBuilder.buildMomentumWorkSet(aDomain, aMaps.mVectorFieldOrdinalsMap, aVariables.vector("previous velocity"), tPrevVelWS->mData);
    aWorkSets.set("previous velocity", tPrevVelWS);

    using PreviousPressureT = typename EvaluationT::PreviousMassScalarType;
    auto tPrevPressWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<PreviousPressureT> > >
        ( Plato::ScalarMultiVectorT<PreviousPressureT>("previous pressure", tNumCells, PhysicsT::SimplexT::mNumMassDofsPerCell) );
    tWorkSetBuilder.buildMassWorkSet(aDomain, aMaps.mScalarFieldOrdinalsMap, aVariables.vector("previous pressure"), tPrevPressWS->mData);
    aWorkSets.set("previous pressure", tPrevPressWS);

    using PreviousTemperatureT = typename EvaluationT::PreviousEnergyScalarType;
    auto tPrevTempWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<PreviousTemperatureT> > >
        ( Plato::ScalarMultiVectorT<PreviousTemperatureT>("previous temperature", tNumCells, PhysicsT::SimplexT::mNumEnergyDofsPerCell) );
    tWorkSetBuilder.buildEnergyWorkSet(aDomain, aMaps.mScalarFieldOrdinalsMap, aVariables.vector("previous temperature"), tPrevTempWS->mData);
    aWorkSets.set("previous temperature", tPrevTempWS);

    using ControlT = typename EvaluationT::ControlScalarType;
    auto tControlWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<ControlT> > >
        ( Plato::ScalarMultiVectorT<ControlT>("control", tNumCells, PhysicsT::SimplexT::mNumNodesPerCell) );
    tWorkSetBuilder.buildControlWorkSet(aDomain, aMaps.mControlOrdinalsMap, aControls, tControlWS->mData);
    aWorkSets.set("control", tControlWS);

    using ConfigT = typename EvaluationT::ConfigScalarType;
    auto tConfig = std::make_shared< Plato::MetaData< Plato::ScalarArray3DT<ConfigT> > >
        ( Plato::ScalarArray3DT<ConfigT>("configuration", tNumCells, PhysicsT::SimplexT::mNumNodesPerCell, PhysicsT::SimplexT::mNumConfigDofsPerNode) );
    tWorkSetBuilder.buildConfigWorkSet(aDomain, aMaps.mNodeCoordinate, tConfig->mData);
    aWorkSets.set("configuration", tConfig);

    auto tCriticalTimeStep = std::make_shared< Plato::MetaData< Plato::ScalarVector > >( Plato::ScalarVector("critical time step", 1) );
    Plato::blas1::copy(aVariables.vector("critical time step"), tCriticalTimeStep->mData);
    aWorkSets.set("critical time step", tCriticalTimeStep);

    if(aVariables.defined("artificial compressibility"))
    {
        auto tArtificialCompressWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVector > >
            ( Plato::ScalarMultiVector("artificial compressibility", tNumCells, PhysicsT::SimplexT::mNumNodesPerCell) );
        Plato::workset_control_scalar_scalar<PhysicsT::SimplexT::mNumNodesPerCell>
            (aDomain, aMaps.mScalarFieldOrdinalsMap, aVariables.vector("artificial compressibility"), tArtificialCompressWS->mData);
        aWorkSets.set("artificial compressibility", tArtificialCompressWS);
    }
}
// function build_vector_function_worksets


/***************************************************************************//**
 * \tparam EvaluationT Forward Automatic Differentiation (FAD) evaluation type
 * \tparam PhysicsT    fluid flow physics type
 *
 * \fn inline void build_vector_function_worksets
 *
 * \brief Build metadata work sets used for the evaluation of a vector function for fluid flow applications.
 *
 * \param [in] aNumCells  total number of cells/elements in the domain
 * \param [in] aControls  1D array of control field
 * \param [in] aVariables state metadata (e.g. pressure, velocity, temperature, etc.)
 * \param [in] aMaps      holds maps from element to local field degrees of freedom
 *
 * \param [in/out] aWorkSets state work sets initialize with the correct FAD type
 ******************************************************************************/
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
    Plato::Fluids::WorkSetBuilder<PhysicsT> tWorkSetBuilder;

    using CurrentPredictorT = typename EvaluationT::MomentumPredictorScalarType;
    auto tPredictorWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<CurrentPredictorT> > >
        ( Plato::ScalarMultiVectorT<CurrentPredictorT>("current predictor", aNumCells, PhysicsT::SimplexT::mNumMomentumDofsPerCell) );
    tWorkSetBuilder.buildMomentumWorkSet(aNumCells, aMaps.mVectorFieldOrdinalsMap, aVariables.vector("current predictor"), tPredictorWS->mData);
    aWorkSets.set("current predictor", tPredictorWS);

    using CurrentVelocityT = typename EvaluationT::CurrentMomentumScalarType;
    auto tCurVelWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<CurrentVelocityT> > >
        ( Plato::ScalarMultiVectorT<CurrentVelocityT>("current velocity", aNumCells, PhysicsT::SimplexT::mNumMomentumDofsPerCell) );
    tWorkSetBuilder.buildMomentumWorkSet(aNumCells, aMaps.mVectorFieldOrdinalsMap, aVariables.vector("current velocity"), tCurVelWS->mData);
    aWorkSets.set("current velocity", tCurVelWS);

    using CurrentPressureT = typename EvaluationT::CurrentMassScalarType;
    auto tCurPressWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<CurrentPressureT> > >
        ( Plato::ScalarMultiVectorT<CurrentPressureT>("current pressure", aNumCells, PhysicsT::SimplexT::mNumMassDofsPerCell) );
    tWorkSetBuilder.buildMassWorkSet(aNumCells, aMaps.mScalarFieldOrdinalsMap, aVariables.vector("current pressure"), tCurPressWS->mData);
    aWorkSets.set("current pressure", tCurPressWS);

    using CurrentTemperatureT = typename EvaluationT::CurrentEnergyScalarType;
    auto tCurTempWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<CurrentTemperatureT> > >
        ( Plato::ScalarMultiVectorT<CurrentTemperatureT>("current temperature", aNumCells, PhysicsT::SimplexT::mNumEnergyDofsPerCell) );
    tWorkSetBuilder.buildEnergyWorkSet(aNumCells, aMaps.mScalarFieldOrdinalsMap, aVariables.vector("current temperature"), tCurTempWS->mData);
    aWorkSets.set("current temperature", tCurTempWS);

    using PreviousVelocityT = typename EvaluationT::PreviousMomentumScalarType;
    auto tPrevVelWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<PreviousVelocityT> > >
        ( Plato::ScalarMultiVectorT<PreviousVelocityT>("previous velocity", aNumCells, PhysicsT::SimplexT::mNumMomentumDofsPerCell) );
    tWorkSetBuilder.buildMomentumWorkSet(aNumCells, aMaps.mVectorFieldOrdinalsMap, aVariables.vector("previous velocity"), tPrevVelWS->mData);
    aWorkSets.set("previous velocity", tPrevVelWS);

    using PreviousPressureT = typename EvaluationT::PreviousMassScalarType;
    auto tPrevPressWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<PreviousPressureT> > >
        ( Plato::ScalarMultiVectorT<PreviousPressureT>("previous pressure", aNumCells, PhysicsT::SimplexT::mNumMassDofsPerCell) );
    tWorkSetBuilder.buildMassWorkSet(aNumCells, aMaps.mScalarFieldOrdinalsMap, aVariables.vector("previous pressure"), tPrevPressWS->mData);
    aWorkSets.set("previous pressure", tPrevPressWS);

    using PreviousTemperatureT = typename EvaluationT::PreviousEnergyScalarType;
    auto tPrevTempWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<PreviousTemperatureT> > >
        ( Plato::ScalarMultiVectorT<PreviousTemperatureT>("previous temperature", aNumCells, PhysicsT::SimplexT::mNumEnergyDofsPerCell) );
    tWorkSetBuilder.buildEnergyWorkSet(aNumCells, aMaps.mScalarFieldOrdinalsMap, aVariables.vector("previous temperature"), tPrevTempWS->mData);
    aWorkSets.set("previous temperature", tPrevTempWS);

    using ControlT = typename EvaluationT::ControlScalarType;
    auto tControlWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<ControlT> > >
        ( Plato::ScalarMultiVectorT<ControlT>("control", aNumCells, PhysicsT::SimplexT::mNumNodesPerCell) );
    tWorkSetBuilder.buildControlWorkSet(aNumCells, aMaps.mControlOrdinalsMap, aControls, tControlWS->mData);
    aWorkSets.set("control", tControlWS);

    using ConfigT = typename EvaluationT::ConfigScalarType;
    auto tConfig = std::make_shared< Plato::MetaData< Plato::ScalarArray3DT<ConfigT> > >
        ( Plato::ScalarArray3DT<ConfigT>("configuration", aNumCells, PhysicsT::SimplexT::mNumNodesPerCell, PhysicsT::SimplexT::mNumConfigDofsPerNode) );
    tWorkSetBuilder.buildConfigWorkSet(aNumCells, aMaps.mNodeCoordinate, tConfig->mData);
    aWorkSets.set("configuration", tConfig);

    auto tCriticalTimeStep = std::make_shared< Plato::MetaData< Plato::ScalarVector > >( Plato::ScalarVector("critical time step", 1) );
    Plato::blas1::copy(aVariables.vector("critical time step"), tCriticalTimeStep->mData);
    aWorkSets.set("critical time step", tCriticalTimeStep);

    if(aVariables.defined("artificial compressibility"))
    {
        auto tArtificialCompressWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVector > >
            ( Plato::ScalarMultiVector("artificial compressibility", aNumCells, PhysicsT::SimplexT::mNumNodesPerCell) );
        Plato::workset_control_scalar_scalar<PhysicsT::SimplexT::mNumNodesPerCell>
            (aNumCells, aMaps.mScalarFieldOrdinalsMap, aVariables.vector("artificial compressibility"), tArtificialCompressWS->mData);
        aWorkSets.set("artificial compressibility", tArtificialCompressWS);
    }
}
// function build_vector_function_worksets



/***************************************************************************//**
 * \tparam PhysicsT    Plato physics type
 * \tparam EvaluationT Forward Automatic Differentiation (FAD) evaluation type
 *
 * \class AbstractScalarFunction
 *
 * \brief Base pure virtual class for Plato scalar functions.
 ******************************************************************************/
template<typename PhysicsT, typename EvaluationT>
class AbstractScalarFunction
{
private:
    using ResultT = typename EvaluationT::ResultScalarType; /*!< result FAD type */

public:
    AbstractScalarFunction(){}
    virtual ~AbstractScalarFunction(){}

    virtual std::string name() const = 0;
    virtual void evaluate(const Plato::WorkSets & aWorkSets, Plato::ScalarVectorT<ResultT> & aResult) const = 0;
    virtual void evaluateBoundary(const Plato::WorkSets & aWorkSets, Plato::ScalarVectorT<ResultT> & aResult) const = 0;
};
// class AbstractScalarFunction


/***************************************************************************//**
 * \tparam PhysicsT    Plato physics type
 * \tparam EvaluationT Forward Automatic Differentiation (FAD) evaluation type
 *
 * \class AverageSurfacePressure
 *
 * \brief Includes functionalities to evaluate the average surface pressure
 *   along a set of user-provided surfaces.
 *
 *                  \f[ \int_{\Gamma} p^n d\Gamma \f],
 *
 * where \f$ n \f$ denotes the current time step and \f$ p \f$ denotes pressure.
 ******************************************************************************/
template<typename PhysicsT, typename EvaluationT>
class AverageSurfacePressure : public Plato::Fluids::AbstractScalarFunction<PhysicsT, EvaluationT>
{
private:
    static constexpr auto mNumSpatialDims       = PhysicsT::SimplexT::mNumSpatialDims;         /*!< number of spatial dimensions */
    static constexpr auto mNumSpatialDimsOnFace = PhysicsT::SimplexT::mNumSpatialDimsOnFace;   /*!< number of spatial dimensions on face */
    static constexpr auto mNumNodesPerFace      = PhysicsT::SimplexT::mNumNodesPerFace;        /*!< number of nodes per face */
    static constexpr auto mNumPressDofsPerNode  = PhysicsT::SimplexT::mNumMassDofsPerNode;     /*!< number of pressure dofs per node */

    using ResultT   = typename EvaluationT::ResultScalarType;      /*!< result FAD type */
    using ConfigT   = typename EvaluationT::ConfigScalarType;      /*!< configuration FAD type */
    using PressureT = typename EvaluationT::CurrentMassScalarType; /*!< pressure FAD type */

    // set local typenames
    using CubatureRule  = Plato::LinearTetCubRuleDegreeOne<mNumSpatialDimsOnFace>; /*!< local short name for cubature rule class */

    // member metadata
    Plato::DataMap& mDataMap; /*!< holds output metadata */
    CubatureRule mCubatureRule; /*!< cubature integration rule */
    const Plato::SpatialDomain& mSpatialDomain; /*!< holds mesh and entity sets metadata for a domain (i.e. element block) */

    // member parameters
    std::string mFuncName; /*!< scalar funciton name */
    std::vector<std::string> mSideSets; /*!< sideset names corresponding to the surfaces associated with the surface integral */

public:
    /***************************************************************************//**
     * \brief Constructor
     * \param [in] aName    scalar function name
     * \param [in] aDomain  holds mesh and entity sets (e.g. node and side sets) metadata
     * \param [in] aDataMap holds output metadata
     * \param [in] aInputs  input file metadata
     ******************************************************************************/
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

    /***************************************************************************//**
     * \brief Destructor
     ******************************************************************************/
    virtual ~AverageSurfacePressure(){}

    /***************************************************************************//**
     * \fn std::string name
     * \brief Returns scalar function name
     * \return scalar function name
     ******************************************************************************/
    std::string name() const override { return mFuncName; }

    /***************************************************************************//**
     * \fn void evaluate
     * \brief Evaluate scalar function inside the computational domain \f$ \Omega \f$.
     * \param [in] aWorkSets holds state work sets initialize with correct FAD types
     * \param [in] aResult   1D output work set of size number of cells
     ******************************************************************************/
    void evaluate(const Plato::WorkSets & aWorkSets, Plato::ScalarVectorT<ResultT> & aResult) const override
    { return; }

    /***************************************************************************//**
     * \fn void evaluateBoundary
     * \brief Evaluate scalar function along the computational boudary \f$ d\Gamma \f$.
     * \param [in] aWorkSets holds state work sets initialize with correct FAD types
     * \param [in] aResult   1D output work set of size number of cells
     ******************************************************************************/
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
        Plato::CalculateSurfaceArea<mNumSpatialDims> tCalculateSurfaceArea;
        Plato::NodeCoordinate<mNumSpatialDims> tCoords(&(mSpatialDomain.Mesh));
        Plato::CalculateSurfaceJacobians<mNumSpatialDims> tCalculateSurfaceJacobians;
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
                    tCalculateSurfaceJacobians(tCellOrdinal, aFaceI, tLocalNodeOrdinals, tConfigurationWS, tJacobians);
                    tCalculateSurfaceArea(aFaceI, tCubatureWeight, tJacobians, tSurfaceAreaTimesCubWeight);

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


/***************************************************************************//**
 * \tparam PhysicsT    Plato physics type
 * \tparam EvaluationT Forward Automatic Differentiation (FAD) evaluation type
 *
 * \class AverageSurfaceTemperature
 *
 * \brief Includes functionalities to evaluate the average surface temperature
 *   along a set of user-provided surfaces.
 *
 *                  \f[ \int_{\Gamma} T^n d\Gamma \f],
 *
 * where \f$ n \f$ denotes the current time step and \f$ T \f$ denotes temperature.
 ******************************************************************************/
template<typename PhysicsT, typename EvaluationT>
class AverageSurfaceTemperature : public Plato::Fluids::AbstractScalarFunction<PhysicsT, EvaluationT>
{
private:
    static constexpr auto mNumSpatialDims       = PhysicsT::SimplexT::mNumSpatialDims;         /*!< number of spatial dimensions */
    static constexpr auto mNumSpatialDimsOnFace = PhysicsT::SimplexT::mNumSpatialDimsOnFace;   /*!< number of spatial dimensions on face */
    static constexpr auto mNumNodesPerFace      = PhysicsT::SimplexT::mNumNodesPerFace;        /*!< number of nodes per face */
    static constexpr auto mNumPressDofsPerNode  = PhysicsT::SimplexT::mNumMassDofsPerNode;     /*!< number of temperature dofs per node */

    using TempT    = typename EvaluationT::CurrentEnergyScalarType; /*!< temperature FAD type */
    using ResultT  = typename EvaluationT::ResultScalarType;        /*!< result FAD type */
    using ConfigT  = typename EvaluationT::ConfigScalarType;        /*!< configuration FAD type */

    // set local typenames
    using CubatureRule  = Plato::LinearTetCubRuleDegreeOne<mNumSpatialDimsOnFace>; /*!< local short name for cubature rule class */

    // member metadata
    Plato::DataMap& mDataMap; /*!< holds output metadata */
    CubatureRule mCubatureRule; /*!< cubature integration rule */
    const Plato::SpatialDomain& mSpatialDomain; /*!< holds mesh and entity sets metadata for a domain (i.e. element block) */

    // member parameters
    std::string mFuncName; /*!< scalar funciton name */
    std::vector<std::string> mWallSets; /*!< sideset names corresponding to the surfaces associated with the surface integral */

public:
    /***************************************************************************//**
     * \brief Constructor
     * \param [in] aName    scalar function name
     * \param [in] aDomain  holds mesh and entity sets (e.g. node and side sets) metadata
     * \param [in] aDataMap holds output metadata
     * \param [in] aInputs  input file metadata
     ******************************************************************************/
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

    /***************************************************************************//**
     * \brief Destructor
     ******************************************************************************/
    virtual ~AverageSurfaceTemperature(){}

    /***************************************************************************//**
     * \fn std::string name
     * \brief Returns scalar function name
     * \return scalar function name
     ******************************************************************************/
    std::string name() const override { return mFuncName; }

    /***************************************************************************//**
     * \fn void evaluate
     * \brief Evaluate scalar function inside the computational domain \f$ \Omega \f$.
     * \param [in] aWorkSets holds state work sets initialize with correct FAD types
     * \param [in] aResult   1D output work set of size number of cells
     ******************************************************************************/
    void evaluate(const Plato::WorkSets & aWorkSets, Plato::ScalarVectorT<ResultT> & aResult) const override
    { return; }

    /***************************************************************************//**
     * \fn void evaluateBoundary
     * \brief Evaluate scalar function along the computational boudary \f$ d\Gamma \f$.
     * \param [in] aWorkSets holds state work sets initialize with correct FAD types
     * \param [in] aResult   1D output work set of size number of cells
     ******************************************************************************/
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
        Plato::CalculateSurfaceArea<mNumSpatialDims> tCalculateSurfaceArea;
        Plato::NodeCoordinate<mNumSpatialDims> tCoords(&(mSpatialDomain.Mesh));
        Plato::CalculateSurfaceJacobians<mNumSpatialDims> tCalculateSurfaceJacobians;
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
                    tCalculateSurfaceJacobians(tCellOrdinal, aFaceI, tLocalNodeOrdinals, tConfigurationWS, tJacobians);
                    tCalculateSurfaceArea(aFaceI, tCubatureWeight, tJacobians, tSurfaceAreaTimesCubWeight);

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


/***************************************************************************//**
 * \tparam NumNodesPerCell number of nodes per cell (integer)
 * \tparam ControlT control work set Forward Automatic Differentiation (FAD) type
 *
 * \fn DEVICE_TYPE inline ControlT brinkman_penalization
 *
 * \brief Evaluate fictitious material penalty model.
 *
 * \f$  \alpha\frac{\left( 1 - \rho \right)}{1 + \epsilon\rho} \f$
 *
 * where \f$ \alpha \f$ denotes a scalar physical parameter, \f$ \rho \f$ denotes
 * the fictitious density field used to depict the geometry, and \f$ \epsilon \f$
 * is a parameter used to improve the convexity of the Brinkman penalization model.
 *
 * \param [in] aCellOrdinal    element/cell ordinal
 * \param [in] aPhysicalParam  physical parameter to be penalized
 * \param [in] aConvexityParam Brinkman model's convexity parameter
 * \param [in] aControlWS      2D control work set
 *
 * \return penalized physical parameter
 ******************************************************************************/
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
// function brinkman_penalization


/***************************************************************************//**
 * \tparam NumNodesPerCell number of nodes per cell (integer)
 * \tparam NumSpaceDim     number of spatial dimensions (integer)
 * \tparam AViewTypeT      input view Forward Automatic Differentiation (FAD) type
 * \tparam BViewTypeT      input view FAD type
 * \tparam CViewTypeT      input view FAD type
 *
 * \fn DEVICE_TYPE inline void strain_rate
 *
 * \brief Evaluate strain rate.
 *
 * \f[ \frac{1}{2}\left( \frac{\partial u_i}{\partial x_j} + \frac{\partial u_j}{\partial x_i} \right) \f]
 *
 * where \f$ \alpha \f$ denotes a scalar physical parameter, \f$ u_i \f$ denotes the
 * i-th component of the velocity field and \f$ x_i \f$ denotes the i-th coordinate.
 *
 * \param [in] aCellOrdinal element/cell ordinal
 * \param [in] aStateWS     2D view with element state work set
 * \param [in] aGradient    3D view with shape function's derivatives
 * \param [in] aStrainRate  3D view with element strain rate
 ******************************************************************************/
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
// function strain_rate


/***************************************************************************//**
 * \tparam PhysicsT    Plato physics type
 * \tparam EvaluationT Forward Automatic Differentiation (FAD) evaluation type
 *
 * \class InternalDissipationEnergy
 *
 * \brief Includes functionalities to evaluate the internal dissipation energy.
 *
 * \f[ \int_{\Omega_e}\left[ \tau_{ij}(\theta):\tau_{ij}(\theta) + \alpha(\theta)u_i^2 \right] d\Omega_e \f],
 *
 * where \f$\theta\f$ denotes the controls, \f$\alpha\f$ denotes the Brinkman
 * penalization parameter.
 ******************************************************************************/
template<typename PhysicsT, typename EvaluationT>
class InternalDissipationEnergy : public Plato::Fluids::AbstractScalarFunction<PhysicsT, EvaluationT>
{
private:
    static constexpr auto mNumSpatialDims    = PhysicsT::SimplexT::mNumSpatialDims;         /*!< number of spatial dimensions */
    static constexpr auto mNumNodesPerCell   = PhysicsT::SimplexT::mNumNodesPerCell;        /*!< number of nodes per cell */
    static constexpr auto mNumVelDofsPerNode = PhysicsT::SimplexT::mNumMomentumDofsPerCell; /*!< number of velocity dofs per node */

    // local forward automatic differentiation typenames
    using ResultT  = typename EvaluationT::ResultScalarType;           /*!< result FAD type */
    using CurVelT  = typename EvaluationT::CurrentMomentumScalarType;  /*!< current velocity FAD type */
    using ConfigT  = typename EvaluationT::ConfigScalarType;           /*!< configuration FAD type */
    using ControlT = typename EvaluationT::ControlScalarType;          /*!< control FAD type */
    using StrainT  = typename Plato::Fluids::fad_type_t<typename PhysicsT::SimplexT, CurVelT, ConfigT>; /*!< strain rate FAD type */

    // set local typenames
    using CubatureRule  = Plato::LinearTetCubRuleDegreeOne<mNumSpatialDims>; /*!< local short name/notation for cubature rule class */

    // member parameters
    std::string mFuncName; /*!< scalar function name */
    Plato::Scalar mPrNum = 1.0; /*!< Prandtl number */
    Plato::Scalar mDaNum = 1.0; /*!< Darcy number */
    Plato::Scalar mBrinkmanConvexityParam = 0.5; /*!< convexity parameter for Brinkmann penalization model */

    // member metadata
    Plato::DataMap& mDataMap; /*!< holds output metadata */
    CubatureRule mCubatureRule; /*!< cubature integration rule */
    const Plato::SpatialDomain& mSpatialDomain; /*!< holds mesh and entity sets metadata for a domain (i.e. element block) */

public:
    /***************************************************************************//**
     * \brief Constructor
     * \param [in] aName    scalar function name
     * \param [in] aDomain  holds mesh and entity sets (e.g. node and side sets) metadata
     * \param [in] aDataMap holds output metadata
     * \param [in] aInputs  input file metadata
     ******************************************************************************/
    InternalDissipationEnergy
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
        this->setBrinkmannModel(aInputs);
    }

    /***************************************************************************//**
     * \brief Destructor
     ******************************************************************************/
    virtual ~InternalDissipationEnergy(){}

    /***************************************************************************//**
     * \fn std::string name
     * \brief Returns scalar function name
     * \return scalar function name
     ******************************************************************************/
    std::string name() const override { return mFuncName; }

    /***************************************************************************//**
     * \fn void evaluate
     * \brief Evaluate scalar function inside the computational domain \f$ \Omega \f$.
     * \param [in] aWorkSets holds state work sets initialize with correct FAD types
     * \param [in] aResult   1D output work set of size number of cells
     ******************************************************************************/
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
        auto tBrinkConvexParam = mBrinkmanConvexityParam;

        auto tCubWeight = mCubatureRule.getCubWeight();
        auto tBasisFunctions = mCubatureRule.getBasisFunctions();
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
        {
            tComputeGradient(aCellOrdinal, tGradient, tConfigWS, tVolumeTimesWeight);
            tVolumeTimesWeight(aCellOrdinal) *= tCubWeight;

            // calculate deviatoric stress contribution to internal energy
            Plato::Fluids::strain_rate<mNumNodesPerCell, mNumSpatialDims>(aCellOrdinal, tCurVelWS, tGradient, tStrainRate);
            auto tTwoTimesPrNum = static_cast<Plato::Scalar>(2.0) * tPrNum;
            Plato::blas2::scale<mNumSpatialDims, mNumSpatialDims>(aCellOrdinal, tTwoTimesPrNum, tStrainRate, tDevStress);
            Plato::blas2::dot<mNumSpatialDims, mNumSpatialDims>(aCellOrdinal, tDevStress, tDevStress, aResult);

            // calculate fictitious material model (i.e. brinkman model) contribution to internal energy
            auto tPermeability = tPrNum / tDaNum;
            ControlT tPenalizedPermeability =
                Plato::Fluids::brinkman_penalization<mNumNodesPerCell>(aCellOrdinal, tPermeability, tBrinkConvexParam, tControlWS);
            tIntrplVectorField(aCellOrdinal, tBasisFunctions, tCurVelWS, tCurVelGP);
            Plato::blas1::dot<mNumSpatialDims>(aCellOrdinal, tCurVelGP, tCurVelGP, tCurVelDotCurVel);
            aResult(aCellOrdinal) += tPenalizedPermeability * tCurVelDotCurVel(aCellOrdinal);

            // apply gauss weight times volume multiplier
            aResult(aCellOrdinal) *= tVolumeTimesWeight(aCellOrdinal);

        }, "internal energy");
    }

    /***************************************************************************//**
     * \fn void evaluateBoundary
     * \brief Evaluate scalar function along the computational boudary \f$ d\Gamma \f$.
     * \param [in] aWorkSets holds state work sets initialize with correct FAD types
     * \param [in] aResult   1D output work set of size number of cells
     ******************************************************************************/
    void evaluateBoundary(const Plato::WorkSets & aWorkSets, Plato::ScalarVectorT<ResultT> & aResult) const override
    { return; }

private:
    /***************************************************************************//**
     * \fn void setBrinkmannModel
     * \brief Set parameters for the Brinkmann fictitious material penalization model.
     * \param [in] aInputs input file metadata
     ******************************************************************************/
    void setBrinkmannModel(Teuchos::ParameterList & aInputs)
    {
        auto tMyCriterionInputs = aInputs.sublist("Criteria").sublist(mFuncName);
        if(tMyCriterionInputs.isSublist("Penalty Function"))
        {
            auto tPenaltyFuncInputs = tMyCriterionInputs.sublist("Penalty Function");
            mBrinkmanConvexityParam = tPenaltyFuncInputs.get<Plato::Scalar>("Brinkman Convexity Parameter", 0.5);
        }
    }
};
// class InternalDissipationEnergy


/***************************************************************************//**
 * \class CriterionBase
 *
 * This pure virtual class defines the template for scalar functions in the form:
 *
 *    \f[ J = J(\phi, U^k, P^k, T^k, X) \f]
 *
 * It manages the evaluation of the function and corresponding derivatives with
 * respect to control \f$\phi\f$, momentum state \f$ U^k \f$, mass state \f$ P^k \f$,
 * energy state \f$ T^k \f$, and configuration \f$ X \f$ variables.
 ******************************************************************************/
class CriterionBase
{
public:
    virtual ~CriterionBase(){}
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
// class CriterionBase


/***************************************************************************//**
 * \tparam PhysicsT fluid flow physics type
 *
 * \class ScalarFunction
 *
 * This class manages the evaluation of scalar functions in the form:
 *
 *                  \f[ J(\phi, U^k, P^k, T^k, X) \f]
 *
 * and respective partial derivatives with respect to control \f$\phi\f$,
 * momentum state \f$ U^k \f$, mass state \f$ P^k \f$, energy state \f$ T^k \f$,
 * and configuration \f$ X \f$.
 ******************************************************************************/
template<typename PhysicsT>
class ScalarFunction : public Plato::Fluids::CriterionBase
{
private:
    std::string mFuncName; /*!< scalar function name */

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
    using ResidualEvalT     = typename Plato::Fluids::Evaluation<typename PhysicsT::SimplexT>::Residual;        /*!< residual FAD evaluation type */
    using GradConfigEvalT   = typename Plato::Fluids::Evaluation<typename PhysicsT::SimplexT>::GradConfig;      /*!< partial wrt configuration FAD evaluation type */
    using GradControlEvalT  = typename Plato::Fluids::Evaluation<typename PhysicsT::SimplexT>::GradControl;     /*!< partial wrt control FAD evaluation type */
    using GradCurVelEvalT   = typename Plato::Fluids::Evaluation<typename PhysicsT::SimplexT>::GradCurMomentum; /*!< partial wrt current velocity state FAD evaluation type */
    using GradCurTempEvalT  = typename Plato::Fluids::Evaluation<typename PhysicsT::SimplexT>::GradCurEnergy;   /*!< partial wrt current temperature state FAD evaluation type */
    using GradCurPressEvalT = typename Plato::Fluids::Evaluation<typename PhysicsT::SimplexT>::GradCurMass;     /*!< partial wrt current pressure state FAD evaluation type */

    // element scalar functions types
    using ResidualFunc     = std::shared_ptr<Plato::Fluids::AbstractScalarFunction<PhysicsT, ResidualEvalT>>;     /*!< short name/notation for a scalar function of residual FAD evaluation type */
    using GradConfigFunc   = std::shared_ptr<Plato::Fluids::AbstractScalarFunction<PhysicsT, GradConfigEvalT>>;   /*!< short name/notation for a scalar function of partial wrt configuration FAD evaluation type */
    using GradControlFunc  = std::shared_ptr<Plato::Fluids::AbstractScalarFunction<PhysicsT, GradControlEvalT>>;  /*!< short name/notation for a scalar function of partial wrt control FAD evaluation type */
    using GradCurVelFunc   = std::shared_ptr<Plato::Fluids::AbstractScalarFunction<PhysicsT, GradCurVelEvalT>>;   /*!< short name/notation for a scalar function of partial wrt current velocity state FAD evaluation type */
    using GradCurTempFunc  = std::shared_ptr<Plato::Fluids::AbstractScalarFunction<PhysicsT, GradCurTempEvalT>>;  /*!< short name/notation for a scalar function of partial wrt current temperature state FAD evaluation type */
    using GradCurPressFunc = std::shared_ptr<Plato::Fluids::AbstractScalarFunction<PhysicsT, GradCurPressEvalT>>; /*!< short name/notation for a scalar function of partial wrt current pressure state FAD evaluation type */

    // element scalar functions per element block, i.e. domain
    std::unordered_map<std::string, ResidualFunc>     mResidualFuncs; /*!< map from domain (i.e. element block) to scalar function of residual FAD evaluation type */
    std::unordered_map<std::string, GradConfigFunc>   mGradConfigFuncs; /*!< map from domain (i.e. element block) to scalar function of partial wrt configuration FAD evaluation type */
    std::unordered_map<std::string, GradControlFunc>  mGradControlFuncs; /*!< map from domain (i.e. element block) to scalar function of partial wrt control FAD evaluation type */
    std::unordered_map<std::string, GradCurVelFunc>   mGradCurrentVelocityFuncs; /*!< map from domain (i.e. element block) to scalar function of partial wrt current velocity state FAD evaluation type */
    std::unordered_map<std::string, GradCurPressFunc> mGradCurrentPressureFuncs; /*!< map from domain (i.e. element block) to scalar function of partial wrt current pressure state FAD evaluation type */
    std::unordered_map<std::string, GradCurTempFunc>  mGradCurrentTemperatureFuncs; /*!< map from domain (i.e. element block) to scalar function of partial wrt current temperature state FAD evaluation type */

    Plato::DataMap& mDataMap; /*!< holds output metadata */
    const Plato::SpatialModel& mSpatialModel; /*!< holds mesh and entity sets metadata */
    Plato::LocalOrdinalMaps<PhysicsT> mLocalOrdinalMaps; /*!< holds maps from element to local state degree of freedom */

public:
    /***************************************************************************//**
     * \brief Constructor
     * \param [in] aModel   holds mesh and entity sets (e.g. node and side sets) metadata for a computational model
     * \param [in] aDataMap holds output metadata
     * \param [in] aInputs  input file metadata
     * \param [in] aName    scalar function name
     ******************************************************************************/
    ScalarFunction
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

    /***************************************************************************//**
     * \brief Destructor
     ******************************************************************************/
    virtual ~ScalarFunction(){}

    /***************************************************************************//**
     * \fn std::string name
     * \brief Return scalar function name.
     * \return scalar function name
     ******************************************************************************/
    std::string name() const
    {
        return mFuncName;
    }

    /***************************************************************************//**
     * \fn Plato::Scalar value
     * \brief Evaluate scalar function.
     *
     * \param [in] aControls  1D control field
     * \param [in] aVariables holds state fields (e.g. velocity, temperature, pressure, etc.)
     *
     * \return scalar function evaluation
     ******************************************************************************/
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
            Plato::Fluids::build_scalar_function_worksets<ResidualEvalT>
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
            Plato::Fluids::build_scalar_function_worksets<ResidualEvalT>
                (tNumCells, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            Plato::ScalarVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells);
            mResidualFuncs.begin()->second->evaluateBoundary(tInputWorkSets, tResultWS);

            tReturnValue += Plato::local_result_sum<Plato::Scalar>(tNumCells, tResultWS);
        }

        return tReturnValue;
    }

    /***************************************************************************//**
     * \fn Plato::ScalarVector gradientConfig
     * \brief Evaluate partial derivative of scalar function with respect to configuration.
     *
     * \param [in] aControls  1D control field
     * \param [in] aVariables holds state fields (e.g. velocity, temperature, pressure, etc.)
     *
     * \return partial derivative of scalar function with respect to configuration
     ******************************************************************************/
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
            Plato::Fluids::build_scalar_function_worksets<GradConfigEvalT>
                (tDomain, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            auto tName = tDomain.getDomainName();
            Plato::ScalarVectorT<ResultScalarT> tResultWS("cells value", tNumCells);
            mGradConfigFuncs.at(tName)->evaluate(tInputWorkSets, tResultWS);

            Plato::assemble_vector_gradient_fad<mNumNodesPerCell, mNumSpatialDims>
                (tDomain, mLocalOrdinalMaps.mVectorFieldOrdinalsMap, tResultWS, tGradient);
        }

        // evaluate boundary
        {
            Plato::WorkSets tInputWorkSets;
            auto tNumCells = mSpatialModel.Mesh.nelems();
            Plato::Fluids::build_scalar_function_worksets<GradConfigEvalT>
                (tNumCells, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            Plato::ScalarVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells);
            mGradConfigFuncs.begin()->second->evaluateBoundary(tInputWorkSets, tResultWS);

            Plato::assemble_vector_gradient_fad<mNumNodesPerCell, mNumSpatialDims>
                (tNumCells, mLocalOrdinalMaps.mVectorFieldOrdinalsMap, tResultWS, tGradient);
        }

        return tGradient;
    }

    /***************************************************************************//**
     * \fn Plato::ScalarVector gradientControl
     * \brief Evaluate partial derivative of scalar function with respect to control.
     *
     * \param [in] aControls  1D control field
     * \param [in] aVariables holds state fields (e.g. velocity, temperature, pressure, etc.)
     *
     * \return partial derivative of scalar function with respect to control
     ******************************************************************************/
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
            Plato::Fluids::build_scalar_function_worksets<GradControlEvalT>
                (tDomain, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            auto tName = tDomain.getDomainName();
            Plato::ScalarVectorT<ResultScalarT> tResultWS("cells value", tNumCells);
            mGradControlFuncs.at(tName)->evaluate(tInputWorkSets, tResultWS);

            Plato::assemble_vector_gradient_fad<mNumNodesPerCell, mNumControlDofsPerNode>
                (tDomain, mLocalOrdinalMaps.mControlOrdinalsMap, tResultWS, tGradient);
        }

        // evaluate boundary
        {
            Plato::WorkSets tInputWorkSets;
            auto tNumCells = mSpatialModel.Mesh.nelems();
            Plato::Fluids::build_scalar_function_worksets<GradControlEvalT>
                (tNumCells, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            Plato::ScalarVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells);
            mGradControlFuncs.begin()->second->evaluateBoundary(tInputWorkSets, tResultWS);

            Plato::assemble_vector_gradient_fad<mNumNodesPerCell, mNumControlDofsPerNode>
                (tNumCells, mLocalOrdinalMaps.mControlOrdinalsMap, tResultWS, tGradient);
        }

        return tGradient;
    }

    /***************************************************************************//**
     * \fn Plato::ScalarVector gradientCurrentPress
     * \brief Evaluate partial derivative of scalar function with respect to current pressure.
     *
     * \param [in] aControls  1D control field
     * \param [in] aVariables holds state fields (e.g. velocity, temperature, pressure, etc.)
     *
     * \return partial derivative of scalar function with respect to current pressure
     ******************************************************************************/
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
            Plato::Fluids::build_scalar_function_worksets<GradCurPressEvalT>
                (tDomain, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            auto tName = tDomain.getDomainName();
            Plato::ScalarVectorT<ResultScalarT> tResultWS("cells value", tNumCells);
            mGradCurrentPressureFuncs.at(tName)->evaluate(tInputWorkSets, tResultWS);

            Plato::assemble_vector_gradient_fad<mNumNodesPerCell, mNumMassDofsPerNode>
                (tDomain, mLocalOrdinalMaps.mScalarFieldOrdinalsMap, tResultWS, tGradient);
        }

        // evaluate boundary
        {
            Plato::WorkSets tInputWorkSets;
            auto tNumCells = mSpatialModel.Mesh.nelems();
            Plato::Fluids::build_scalar_function_worksets<GradCurPressEvalT>
                (tNumCells, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            Plato::ScalarVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells);
            mGradCurrentPressureFuncs.begin()->second->evaluateBoundary(tInputWorkSets, tResultWS);

            Plato::assemble_vector_gradient_fad<mNumNodesPerCell, mNumMassDofsPerNode>
                (tNumCells, mLocalOrdinalMaps.mScalarFieldOrdinalsMap, tResultWS, tGradient);
        }

        return tGradient;
    }

    /***************************************************************************//**
     * \fn Plato::ScalarVector gradientCurrentTemp
     * \brief Evaluate partial derivative of scalar function with respect to current temperature.
     *
     * \param [in] aControls  1D control field
     * \param [in] aVariables holds state fields (e.g. velocity, temperature, pressure, etc.)
     *
     * \return partial derivative of scalar function with respect to current temperature
     ******************************************************************************/
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
            Plato::Fluids::build_scalar_function_worksets<GradCurTempEvalT>
                (tDomain, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            auto tName = tDomain.getDomainName();
            Plato::ScalarVectorT<ResultScalarT> tResultWS("cells value", tNumCells);
            mGradCurrentTemperatureFuncs.at(tName)->evaluate(tInputWorkSets, tResultWS);

            Plato::assemble_vector_gradient_fad<mNumNodesPerCell, mNumEnergyDofsPerNode>
                (tDomain, mLocalOrdinalMaps.mScalarFieldOrdinalsMap, tResultWS, tGradient);
        }

        // evaluate boundary
        {
            Plato::WorkSets tInputWorkSets;
            auto tNumCells = mSpatialModel.Mesh.nelems();
            Plato::Fluids::build_scalar_function_worksets<GradCurTempEvalT>
                (tNumCells, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            Plato::ScalarVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells);
            mGradCurrentTemperatureFuncs.begin()->second->evaluateBoundary(tInputWorkSets, tResultWS);

            Plato::assemble_vector_gradient_fad<mNumNodesPerCell, mNumEnergyDofsPerNode>
                (tNumCells, mLocalOrdinalMaps.mScalarFieldOrdinalsMap, tResultWS, tGradient);
        }

        return tGradient;
    }

    /***************************************************************************//**
     * \fn Plato::ScalarVector gradientCurrentVel
     * \brief Evaluate partial derivative of scalar function with respect to current velocity.
     *
     * \param [in] aControls  1D control field
     * \param [in] aVariables holds state fields (e.g. velocity, temperature, pressure, etc.)
     *
     * \return partial derivative of scalar function with respect to current velocity
     ******************************************************************************/
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
            Plato::Fluids::build_scalar_function_worksets<GradCurVelEvalT>
                (tDomain, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            auto tName = tDomain.getDomainName();
            Plato::ScalarVectorT<ResultScalarT> tResultWS("cells value", tNumCells);
            mGradCurrentVelocityFuncs.at(tName)->evaluate(tInputWorkSets, tResultWS);

            Plato::assemble_vector_gradient_fad<mNumNodesPerCell, mNumMomentumDofsPerNode>
                (tDomain, mLocalOrdinalMaps.mVectorFieldOrdinalsMap, tResultWS, tGradient);
        }

        // evaluate boundary
        {
            Plato::WorkSets tInputWorkSets;
            auto tNumCells = mSpatialModel.Mesh.nelems();
            Plato::Fluids::build_scalar_function_worksets<GradCurVelEvalT>
                (tNumCells, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            Plato::ScalarVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells);
            mGradCurrentVelocityFuncs.begin()->second->evaluateBoundary(tInputWorkSets, tResultWS);

            Plato::assemble_vector_gradient_fad<mNumNodesPerCell, mNumMomentumDofsPerNode>
                (tNumCells, mLocalOrdinalMaps.mVectorFieldOrdinalsMap, tResultWS, tGradient);
        }

        return tGradient;
    }

private:
    /***************************************************************************//**
     * \fn void initialize
     * \brief Initialize maps from domain name to scalar function based on appropriate FAD evaluation type.
     * \param [in] aInputs input file metadata
     ******************************************************************************/
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
// class ScalarFunction















/***************************************************************************//**
 * \tparam PhysicsT    Plato physics type
 * \tparam EvaluationT Forward Automatic Differentiation (FAD) evaluation type
 *
 * \class AbstractVectorFunction
 *
 * \brief Pure virtual base class for Plato vector functions.
 ******************************************************************************/
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


/***************************************************************************//**
 * \tparam NumNodes number of nodes in cell/element (integer)
 * \tparam SpaceDim spatial dimensions (integer)
 * \tparam ResultT  output work set Forward Automatic Differentiation (FAD) type
 * \tparam ConfigT  configuration work set FAD type
 * \tparam PrevVelT previous velocity work set FAD type
 *
 * \fn device_type inline void calculate_advected_momentum_forces
 *
 * \brief Calculate advection momentum forces, defined as
 *
 * \f[
 *   \alpha\frac{\partial \bar{u}_j^n}{\partial\bar{x}_j}\bar{u}_i^n + \bar{u}_j^n \frac{\partial \bar{u}_i^n}{\partial\bar{x}_j}
 * \f]
 *
 * where \f$\alpha\f$ is a scalar multiplier, \f$ u_i \f$ is the i-th velocity
 * field, \f$ x_i \f$ is the i-th coordinate.
 *
 ******************************************************************************/
template
<Plato::OrdinalType NumNodes,
 Plato::OrdinalType SpaceDim,
 typename ResultT,
 typename ConfigT,
 typename PrevVelT>
DEVICE_TYPE inline void
calculate_advected_momentum_forces
(const Plato::OrdinalType & aCellOrdinal,
 const Plato::ScalarArray3DT<ConfigT> & aGradient,
 const Plato::ScalarMultiVectorT<PrevVelT> & aPrevVelWS,
 const Plato::ScalarMultiVectorT<PrevVelT> & aPrevVelGP,
 const Plato::ScalarMultiVectorT<ResultT> & aResult,
 Plato::Scalar aMultiplier = 1.0)
{
    for(Plato::OrdinalType tNode = 0; tNode < NumNodes; tNode++)
    {
        for(Plato::OrdinalType tDimI = 0; tDimI < SpaceDim; tDimI++)
        {
            auto tCellDofI = (SpaceDim * tNode) + tDimI;
            for(Plato::OrdinalType tDimJ = 0; tDimJ < SpaceDim; tDimJ++)
            {
                aResult(aCellOrdinal, tDimI) += aMultiplier * ( aPrevVelGP(aCellOrdinal, tDimJ) *
                    ( aGradient(aCellOrdinal, tNode, tDimJ) * aPrevVelWS(aCellOrdinal, tCellDofI) ) );
            }
        }
    }
}
// function calculate_advected_momentum_forces

/***************************************************************************//**
 * \tparam NumNodes number of nodes in cell/element (integer)
 * \tparam SpaceDim spatial dimensions (integer)
 * \tparam ResultT  output work set Forward Automatic Differentiation (FAD) type
 * \tparam ConfigT  configuration work set FAD type
 * \tparam ControlT control work set FAD type
 * \tparam StrainT  strain rate work set FAD type
 *
 * \fn device_type inline void integrate_viscous_forces
 *
 * \brief Integrate viscous forces, which are defined as
 *
 * \f[
 * \alpha\int_{\Omega}\frac{\partial w_i^h}{\partial\bar{x}_j}\bar\tau_{ij}^n\,d\Omega
 * \f]
 *
 * where the deviatoric stress tensor \f$\bar\tau_{ij}^n\f$ and \f$\alpha\f$
 * denotes a scalar multiplier.
 *
 ******************************************************************************/
template
<Plato::OrdinalType NumNodes,
 Plato::OrdinalType SpaceDim,
 typename ResultT,
 typename ConfigT,
 typename ControlT,
 typename StrainT>
DEVICE_TYPE inline void
integrate_viscous_forces
(const Plato::OrdinalType & aCellOrdinal,
 const ControlT & aPrandtlNumber,
 const Plato::ScalarVectorT<ConfigT> & aCellVolume,
 const Plato::ScalarArray3DT<ConfigT> & aGradient,
 const Plato::ScalarArray3DT<StrainT> & aStrainRate,
 const Plato::ScalarMultiVectorT<ResultT> & aResult,
 Plato::Scalar aMultiplier = 1.0)
{
    for(Plato::OrdinalType tNode = 0; tNode < NumNodes; tNode++)
    {
        for(Plato::OrdinalType tDimI = 0; tDimI < SpaceDim; tDimI++)
        {
            auto tDofIndex = (SpaceDim * tNode) + tDimI;
            for(Plato::OrdinalType tDimJ = 0; tDimJ < SpaceDim; tDimJ++)
            {
                aResult(aCellOrdinal, tDofIndex) += aMultiplier * aCellVolume(aCellOrdinal) * aGradient(aCellOrdinal, tNode, tDimJ)
                    * ( static_cast<Plato::Scalar>(2.0) * aPrandtlNumber * aStrainRate(aCellOrdinal, tDimI, tDimJ) );
            }
        }
    }
}
// function integrate_viscous_forces

/***************************************************************************//**
 * \tparam SpaceDim spatial dimensions (integer)
 * \tparam ResultT   output work set Forward Automatic Differentiation (FAD) type
 * \tparam ControlT  control work set FAD type
 * \tparam PrevTempT previous temperature work set FAD type
 *
 * \fn device_type inline void calculate_natural_convective_forces
 *
 * \brief Calculate natural convective forces, which are defined as
 *
 * \f[
 * \alpha Gr_i Pr^2\bar{T}^n
 * \f]
 *
 * where \f$\alpha\f$ denotes a scalar multiplier.
 *
 ******************************************************************************/
template
<Plato::OrdinalType SpaceDim,
 typename ResultT,
 typename ControlT,
 typename PrevTempT>
DEVICE_TYPE inline void
calculate_natural_convective_forces
(const Plato::OrdinalType & aCellOrdinal,
 const ControlT & aPrTimesPr,
 const Plato::ScalarVector & aGrashofNum,
 const Plato::ScalarVectorT<PrevTempT> & aPrevTempGP,
 const Plato::ScalarMultiVectorT<ResultT> & aResult,
 Plato::Scalar aMultiplier = 1.0)
{
    for (Plato::OrdinalType tDim = 0; tDim < SpaceDim; tDim++)
    {
        aResult(aCellOrdinal, tDim) += aMultiplier * aGrashofNum(tDim) * aPrTimesPr * aPrevTempGP(aCellOrdinal);
    }
}
// function calculate_natural_convective_forces


/***************************************************************************//**
 * \tparam SpaceDim spatial dimensions (integer)
 * \tparam ResultT   output work set Forward Automatic Differentiation (FAD) type
 * \tparam ControlT  control work set FAD type
 * \tparam PrevVelT  previous velocity work set FAD type
 *
 * \fn device_type inline void calculate_brinkman_forces
 *
 * \brief Calculate Brinkmann forces, defined as
 *
 * \f[
 * \alpha\frac{Pr}{Da} u^{n-1}_i
 * \f]
 *
 * where \f$\alpha\f$ is a scalar multiplier, \f$Pr\f$ is the Prandtl number,
 * \f$Da\f$ is the Darcy number, and \f$u_i^{n-1}\f$ is i-th component of the
 * previous velocity field.
 *
 ******************************************************************************/
template
<Plato::OrdinalType SpaceDim,
 typename ResultT,
 typename ControlT,
 typename PrevVelT>
DEVICE_TYPE inline void
calculate_brinkman_forces
(const Plato::OrdinalType & aCellOrdinal,
 const ControlT & aPenalizedBrinkmanCoeff,
 const Plato::ScalarMultiVectorT<PrevVelT> & aPrevVelGP,
 const Plato::ScalarMultiVectorT<ResultT> & aResult,
 Plato::Scalar aMultiplier = 1.0)
{
    for (Plato::OrdinalType tDim = 0; tDim < SpaceDim; tDim++)
    {
        aResult(aCellOrdinal, tDim) += aMultiplier * aPenalizedBrinkmanCoeff * aPrevVelGP(aCellOrdinal, tDim);
    }
}
// function calculate_brinkman_forces


/***************************************************************************//**
 * \tparam NumNodes number of nodes in cell/element (integer)
 * \tparam SpaceDim   spatial dimensions (integer)
 * \tparam ResultT    output work set Forward Automatic Differentiation (FAD) type
 * \tparam ConfigT    configuration work set FAD type
 * \tparam PrevVelT   previous velocity work set FAD type
 * \tparam StabilityT stabilizing force work set at quadrature points FAD type
 *
 * \fn device_type inline void integrate_stabilizing_vector_force
 *
 * \brief Integrate stabilizing momentum forces, defined as
 *
 * \f[
 *   \alpha\int_{\Omega} \left( \frac{\partial w_i^h}{\partial\bar{x}_k}\bar{u}^n_k \right) \hat{R}^n_{\bar{u}_i}\, d\Omega
 * \f]
 *
 * where \f$\alpha\f$ denotes a scalar multiplier and \f$\hat{R}^n_{\bar{u}_i}\f$
 * is the stabilizing residual.
 *
 ******************************************************************************/
template
<Plato::OrdinalType NumNodes,
 Plato::OrdinalType SpaceDim,
 typename ResultT,
 typename ConfigT,
 typename PrevVelT,
 typename StabilityT>
DEVICE_TYPE inline void
integrate_stabilizing_vector_force
(const Plato::OrdinalType & aCellOrdinal,
 const Plato::ScalarVectorT<ConfigT> & aCellVolume,
 const Plato::ScalarArray3DT<ConfigT> & aGradient,
 const Plato::ScalarMultiVectorT<PrevVelT> & aPrevVelGP,
 const Plato::ScalarMultiVectorT<StabilityT> & aStabilization,
 const Plato::ScalarMultiVectorT<ResultT> & aResult,
 Plato::Scalar aMultiplier = 1.0)
{
    for(Plato::OrdinalType tNode = 0; tNode < NumNodes; tNode++)
    {
        for(Plato::OrdinalType tDimI = 0; tDimI < SpaceDim; tDimI++)
        {
            auto tLocalCellDof = (SpaceDim * tNode) + tDimI;
            for(Plato::OrdinalType tDimK = 0; tDimK < SpaceDim; tDimK++)
            {
                aResult(aCellOrdinal, tLocalCellDof) += aMultiplier * ( aGradient(aCellOrdinal, tNode, tDimK) *
                    ( aPrevVelGP(aCellOrdinal, tDimK) * aStabilization(aCellOrdinal, tDimI) ) ) * aCellVolume(aCellOrdinal);
            }
        }
    }
}
// function integrate_stabilizing_vector_force

/***************************************************************************//**
 * \tparam NumNodes number of nodes in cell/element (integer)
 * \tparam SpaceDim spatial dimensions (integer)
 * \tparam ResultT  output work set Forward Automatic Differentiation (FAD) type
 * \tparam ConfigT  configuration work set FAD type
 * \tparam PredVelT previous velocity work set FAD type
 * \tparam PrevVelT predicted velocity work set FAD type
 *
 * \fn device_type inline void integrate_momentum_inertial_forces
 *
 * \brief Integrate momentum inertial forces, defined as
 *
 * Predictor Step:
 * \f[
 *   \alpha\int_{\Omega} w_i^h\left(\bar{u}^{\ast}_i - \bar{u}_i^{n}\right) d\Omega
 * \f]
 *
 * or
 *
 * Corrector Step:
 * \f[
 *   \alpha\int_{\Omega} w_i^h\left(\bar{u}_i^{n+1} - \bar{u}^{\ast}_i\right) d\Omega
 * \f]
 *
 * where \f$\alpha\f$ denotes a scalar multiplier.
 *
 ******************************************************************************/
template
<Plato::OrdinalType NumNodes,
 Plato::OrdinalType SpaceDim,
 typename ResultT,
 typename ConfigT,
 typename PredVelT,
 typename PrevVelT>
DEVICE_TYPE inline void
integrate_momentum_inertial_forces
(const Plato::OrdinalType & aCellOrdinal,
 const Plato::ScalarVector & aBasisFunctions,
 const Plato::ScalarVectorT<ConfigT> & aCellVolume,
 const Plato::ScalarMultiVectorT<PredVelT> & aVecA_GP,
 const Plato::ScalarMultiVectorT<PrevVelT> & aVecB_GP,
 const Plato::ScalarMultiVectorT<ResultT> & aResult)
{
    for(Plato::OrdinalType tNode = 0; tNode < NumNodes; tNode++)
    {
        for(Plato::OrdinalType tDimI = 0; tDimI < SpaceDim; tDimI++)
        {
            auto tDofIndex = (SpaceDim * tNode) + tDimI;
            aResult(aCellOrdinal, tDofIndex) += aCellVolume(aCellOrdinal) * aBasisFunctions(tNode) *
                ( aVecB_GP(aCellOrdinal, tDimI) );
                //( aVecA_GP(aCellOrdinal, tDimI) - aVecB_GP(aCellOrdinal, tDimI) );
        }
    }
}
// function integrate_momentum_inertial_forces

/***************************************************************************//**
 * \tparam NumNodesPerCell number of nodes in cell/element (integer)
 * \tparam NumDofsPerNode  number of degrees of freedom per node (integer)
 * \tparam ResultT         output work set Forward Automatic Differentiation (FAD) type
 * \tparam ConfigT         configuration work set FAD type
 * \tparam FieldT          vector field work set FAD type
 *
 * \fn device_type inline void integrate_vector_field
 *
 * \brief Integrate vector field, defined as
 *
 * \f[
 *   \alpha\int_{\Omega} w_i^h f_i d\Omega
 * \f]
 *
 * where \f$\alpha\f$ denotes a scalar multiplier.
 *
 ******************************************************************************/
template
<Plato::OrdinalType NumNodesPerCell,
 Plato::OrdinalType NumDofsPerNode,
 typename ResultT,
 typename ConfigT,
 typename FieldT>
DEVICE_TYPE inline
void integrate_vector_field
(const Plato::OrdinalType & aCellOrdinal,
 const Plato::ScalarVector & aBasisFunctions,
 const Plato::ScalarVectorT<ConfigT> & aCellVolume,
 const Plato::ScalarMultiVectorT<FieldT> & aField,
 const Plato::ScalarMultiVectorT<ResultT> & aResult,
 Plato::Scalar aMultiplier = 1.0)
{
    for(Plato::OrdinalType tNode = 0; tNode < NumNodesPerCell; tNode++)
    {
        for(Plato::OrdinalType tDof = 0; tDof < NumDofsPerNode; tDof++)
        {
            auto tLocalCellDof = (NumDofsPerNode * tNode) + tDof;
            aResult(aCellOrdinal, tLocalCellDof) += aMultiplier * aCellVolume(aCellOrdinal) *
                aBasisFunctions(tNode) * aField(aCellOrdinal, tDof);
        }
    }
}
// function integrate_vector_field


/***************************************************************************//**
 * \fn inline bool is_dimensionless_parameter_defined
 *
 * \brief Check if dimensionless parameter is deifned.
 *
 * \param [in] aTag    parameter tag
 * \param [in] aInputs input file metadata
 *
 * \return boolean (true or false)
 ******************************************************************************/
inline bool is_dimensionless_parameter_defined
(const std::string & aTag,
 Teuchos::ParameterList & aInputs)
{
    if(aInputs.isSublist("Hyperbolic") == false)
    {
        THROWERR("'Hyperbolic' Parameter List is not defined.")
    }

    auto tHyperbolic = aInputs.sublist("Hyperbolic");
    if( tHyperbolic.isSublist("Dimensionless Properties") == false )
    {
        THROWERR("'Dimensionless Properties' sublist is not defined.")
    }
    auto tSublist = tHyperbolic.sublist("Dimensionless Properties");
    auto tIsDefined = tSublist.isParameter(aTag); 
    return tIsDefined;
}
// function is_dimensionless_parameter_defined


/***************************************************************************//**
 * \fn inline Plato::Scalar dimensionless_reynolds_number
 *
 * \brief Parse Reynolds number from input file.
 * \param [in] aInputs input file metadata
 * \return Reynolds number
 ******************************************************************************/
inline Plato::Scalar
dimensionless_reynolds_number
(Teuchos::ParameterList & aInputs)
{
    if(aInputs.isSublist("Hyperbolic") == false)
    {
        THROWERR("'Hyperbolic' Parameter List is not defined.")
    }
    auto tHyperbolic = aInputs.sublist("Hyperbolic");
    auto tReNum = Plato::parse_parameter<Plato::Scalar>("Reynolds Number", "Dimensionless Properties", tHyperbolic);
    return tReNum;
}
// function dimensionless_reynolds_number

/***************************************************************************//**
 * \fn inline Plato::Scalar dimensionless_prandtl_number
 *
 * \brief Parse Prandtl number from input file.
 * \param [in] aInputs input file metadata
 * \return Prandtl number
 ******************************************************************************/
inline Plato::Scalar
dimensionless_prandtl_number
(Teuchos::ParameterList & aInputs)
{
    if(aInputs.isSublist("Hyperbolic") == false)
    {
        THROWERR("'Hyperbolic' Parameter List is not defined.")
    }
    auto tHyperbolic = aInputs.sublist("Hyperbolic");
    auto tPrNum = Plato::parse_parameter<Plato::Scalar>("Prandtl Number", "Dimensionless Properties", tHyperbolic);
    return tPrNum;
}
// function dimensionless_prandtl_number


/***************************************************************************//**
 * \fn inline std::string heat_transfer_tag
 *
 * \brief Parse heat transfer mechanism tag from input file.
 * \param [in] aInputs input file metadata
 * \return heat transfer mechanism tag
 ******************************************************************************/
inline std::string heat_transfer_tag
(Teuchos::ParameterList & aInputs)
{
    if(aInputs.isSublist("Hyperbolic") == false)
    {
        THROWERR("'Hyperbolic' Parameter List is not defined.")
    }
    auto tHyperbolic = aInputs.sublist("Hyperbolic");
    auto tTag = tHyperbolic.get<std::string>("Heat Transfer", "None");
    auto tHeatTransfer = Plato::tolower(tTag);
    return tHeatTransfer;
}
// function heat_transfer_tag

/***************************************************************************//**
 * \fn inline bool calculate_heat_transfer
 *
 * \brief Returns true if buoyancy forces are part of the calculations, else, returns false.
 * \param [in] aInputs input file metadata
 * \return boolean (true or false)
 ******************************************************************************/
inline bool calculate_heat_transfer
(Teuchos::ParameterList & aInputs)
{   
    auto tHeatTransfer = Plato::Fluids::heat_transfer_tag(aInputs);
    auto tCalculateHeatTransfer = tHeatTransfer == "none" ? false : true;
    return tCalculateHeatTransfer;
}
// function calculate_heat_transfer

/***************************************************************************//**
 * \fn inline Plato::Scalar dimensionless_viscosity_constant
 *
 * \brief Parse dimensionless viscocity \f$ \nu f\$, where \f$ \nu=\frac{1}{Re} f\$
 * if forced convection dominates and \f$ \nu=Pr \f$ is natural convection dominates.
 *
 * \param [in] aInputs input file metadata
 * \return dimensionless viscocity
 ******************************************************************************/
inline Plato::Scalar
dimensionless_viscosity_constant
(Teuchos::ParameterList & aInputs)
{
    auto tHeatTransfer = Plato::Fluids::heat_transfer_tag(aInputs);
    if(tHeatTransfer == "forced" || tHeatTransfer == "mixed" || tHeatTransfer == "none")
    {
        auto tReNum = Plato::Fluids::dimensionless_reynolds_number(aInputs);
        auto tViscocity = static_cast<Plato::Scalar>(1) / tReNum;
        return tViscocity;
    }
    else if(tHeatTransfer == "natural")
    {
        auto tViscocity = Plato::Fluids::dimensionless_prandtl_number(aInputs);
        return tViscocity;
    }
    else
    {
        THROWERR(std::string("'Heat Transfer' mechanism with tag '") + tHeatTransfer + "' is not supported.")
    }
}
// function dimensionless_viscosity_constant


/***************************************************************************//**
 * \fn inline Plato::Scalar dimensionless_natural_convection_buoyancy_constant
 *
 * \brief Parse buoyancy constant for natural convection problems.
 *
 * \param [in] aInputs input file metadata
 *
 * \return dimensionless buoyancy constant
 ******************************************************************************/
inline Plato::Scalar
dimensionless_natural_convection_buoyancy_constant
(Teuchos::ParameterList & aInputs)
{
    auto tPrNum = Plato::Fluids::dimensionless_prandtl_number(aInputs);
    if(Plato::Fluids::is_dimensionless_parameter_defined("Rayleigh Number", aInputs))
    {
        auto tBuoyancy = tPrNum;
        return tBuoyancy;
    }
    else if(Plato::Fluids::is_dimensionless_parameter_defined("Grashof Number", aInputs))
    {
        auto tBuoyancy = tPrNum*tPrNum;
        return tBuoyancy;
    }
    else
    {
        THROWERR("Natural convection properties are not defined. One of these two options should be provided: 'Grashof Number' or 'Rayleigh Number'")
    }
}
// function dimensionless_natural_convection_buoyancy_constant


/***************************************************************************//**
 * \fn inline Plato::Scalar dimensionless_mixed_convection_buoyancy_constant
 *
 * \brief Parse buoyancy constant for mixed convection problems.
 *
 * \param [in] aInputs input file metadata
 *
 * \return dimensionless buoyancy constant
 ******************************************************************************/
inline Plato::Scalar
dimensionless_mixed_convection_buoyancy_constant
(Teuchos::ParameterList & aInputs)
{
    if(Plato::Fluids::is_dimensionless_parameter_defined("Richardson Number", aInputs))
    {
        return static_cast<Plato::Scalar>(1.0);
    }
    else if(Plato::Fluids::is_dimensionless_parameter_defined("Grashof Number", aInputs))
    {
        auto tReNum = Plato::Fluids::dimensionless_reynolds_number(aInputs);
        auto tBuoyancy = static_cast<Plato::Scalar>(1.0) / (tReNum * tReNum);
        return tBuoyancy;
    }
    else
    {
        THROWERR("Mixed convection properties are not defined. One of these two options should be provided: 'Grashof Number' or 'Richardson Number'")
    }
}
// function dimensionless_mixed_convection_buoyancy_constant

/***************************************************************************//**
 * \fn inline Plato::Scalar dimensionless_buoyancy_constant
 *
 * \brief Parse dimensionless buoyancy constant \f$ \beta f\$, where \f$ \beta=
 * \frac{1}{Re^2} f\$ if forced convection dominates. The buoyancy constant for
 * natural convection dominated problems is given by \f$ \nu=Pr^2 \f$ or \f$ \nu=Pr \f$
 * based on the dimensionless convective constant (i.e. Rayleigh or Grashof number)
 * provided by the user.
 *
 * \param [in] aInputs input file metadata
 *
 * \return dimensionless buoyancy constant
 ******************************************************************************/
inline Plato::Scalar
dimensionless_buoyancy_constant
(Teuchos::ParameterList & aInputs)
{
    Plato::Scalar tBuoyancy = 0.0; // heat transfer calculations inactive if buoyancy = 0.0

    auto tHeatTransfer = Plato::Fluids::heat_transfer_tag(aInputs);
    if(tHeatTransfer == "mixed")
    {
        tBuoyancy = Plato::Fluids::dimensionless_mixed_convection_buoyancy_constant(aInputs);
    }
    else if(tHeatTransfer == "natural")
    {
        tBuoyancy = Plato::Fluids::dimensionless_natural_convection_buoyancy_constant(aInputs);
    }
    else if(tHeatTransfer == "forced" || tHeatTransfer == "none")
    {
        tBuoyancy = 0.0;
    }
    else
    {
        THROWERR(std::string("'Heat Transfer' mechanism with tag '") + tHeatTransfer + "' is not supported.")
    }

    return tBuoyancy;
}
// function dimensionless_buoyancy_constant


/***************************************************************************//**
 * \tparam SpaceDim spatial dimensions (integer)
 *
 * \fn inline Plato::ScalarVector dimensionless_rayleigh_number
 *
 * \brief Parse array of dimensionless Rayleigh constants.
 *
 * \param [in] aInputs input file metadata
 * \return Rayleigh constants
 ******************************************************************************/
template<Plato::OrdinalType SpaceDim>
inline Plato::ScalarVector
dimensionless_rayleigh_number
(Teuchos::ParameterList & aInputs)
{
    if(aInputs.isSublist("Hyperbolic") == false)
    {
        THROWERR("'Hyperbolic' Parameter List is not defined.")
    }

    auto tHyperbolic = aInputs.sublist("Hyperbolic");
    auto tTag = tHyperbolic.get<std::string>("Heat Transfer", "None");
    auto tHeatTransfer = Plato::tolower(tTag);
    auto tCalculateHeatTransfer = tHeatTransfer == "none" ? false : true;

    Plato::ScalarVector tOuput("Rayleigh Number", SpaceDim);
    if(tCalculateHeatTransfer)
    {
        auto tRaNum = Plato::parse_parameter<Teuchos::Array<Plato::Scalar>>("Rayleigh Number", "Dimensionless Properties", tHyperbolic);
        if(tRaNum.size() != SpaceDim)
        {
            THROWERR(std::string("'Rayleigh Number' array length should match the number of spatial dimensions. ")
                + "Array length is '" + std::to_string(tRaNum.size()) + "' and the number of spatial dimensions is '"
                + std::to_string(SpaceDim) + "'.")
        }

        auto tHostRaNum = Kokkos::create_mirror(tOuput);
        for(Plato::OrdinalType tDim = 0; tDim < SpaceDim; tDim++)
        {
            tHostRaNum(tDim) = tRaNum[tDim];
        }
        Kokkos::deep_copy(tOuput, tHostRaNum);
    }
    else
    {
        Plato::blas1::fill(0.0, tOuput);
    }

    return tOuput;
}
// function dimensionless_rayleigh_number

/***************************************************************************//**
 * \tparam SpaceDim spatial dimensions (integer)
 *
 * \fn inline Plato::ScalarVector dimensionless_grashof_number
 *
 * \brief Parse array of dimensionless Grashof constants.
 *
 * \param [in] aInputs input file metadata
 * \return Grashof constants
 ******************************************************************************/
template<Plato::OrdinalType SpaceDim>
inline Plato::ScalarVector
dimensionless_grashof_number
(Teuchos::ParameterList & aInputs)
{
    if(aInputs.isSublist("Hyperbolic") == false)
    {
        THROWERR("'Hyperbolic' Parameter List is not defined.")
    }

    auto tHyperbolic = aInputs.sublist("Hyperbolic");
    auto tTag = tHyperbolic.get<std::string>("Heat Transfer", "None");
    auto tHeatTransfer = Plato::tolower(tTag);
    auto tCalculateHeatTransfer = tHeatTransfer == "none" ? false : true;

    Plato::ScalarVector tOuput("Grashof Number", SpaceDim);
    if(tCalculateHeatTransfer)
    {
        auto tGrNum = Plato::parse_parameter<Teuchos::Array<Plato::Scalar>>("Grashof Number", "Dimensionless Properties", tHyperbolic);
        if(tGrNum.size() != SpaceDim)
        {
            THROWERR(std::string("'Grashof Number' array length should match the number of spatial dimensions. ")
                + "Array length is '" + std::to_string(tGrNum.size()) + "' and the number of spatial dimensions is '"
                + std::to_string(SpaceDim) + "'.")
        }

        auto tHostGrNum = Kokkos::create_mirror(tOuput);
        for(Plato::OrdinalType tDim = 0; tDim < SpaceDim; tDim++)
        {
            tHostGrNum(tDim) = tGrNum[tDim];
        }
        Kokkos::deep_copy(tOuput, tHostGrNum);
    }
    else
    {
        Plato::blas1::fill(0.0, tOuput);
    }

    return tOuput;
}
// function dimensionless_grashof_number

/***************************************************************************//**
 * \tparam SpaceDim spatial dimensions (integer)
 *
 * \fn inline Plato::ScalarVector dimensionless_richardson_number
 *
 * \brief Parse array of dimensionless Richardson constants.
 *
 * \param [in] aInputs input file metadata
 * \return Richardson constants
 ******************************************************************************/
template<Plato::OrdinalType SpaceDim>
inline Plato::ScalarVector
dimensionless_richardson_number
(Teuchos::ParameterList & aInputs)
{
    if(aInputs.isSublist("Hyperbolic") == false)
    {
        THROWERR("'Hyperbolic' Parameter List is not defined.")
    }

    auto tHyperbolic = aInputs.sublist("Hyperbolic");
    auto tTag = tHyperbolic.get<std::string>("Heat Transfer", "None");
    auto tHeatTransfer = Plato::tolower(tTag);
    auto tCalculateHeatTransfer = tHeatTransfer == "none" ? false : true;

    Plato::ScalarVector tOuput("Grashof Number", SpaceDim);
    if(tCalculateHeatTransfer)
    {
        auto tRiNum = Plato::parse_parameter<Teuchos::Array<Plato::Scalar>>("Richardson Number", "Dimensionless Properties", tHyperbolic);
        if(tRiNum.size() != SpaceDim)
        {
            THROWERR(std::string("'Richardson Number' array length should match the number of spatial dimensions. ")
                + "Array length is '" + std::to_string(tRiNum.size()) + "' and the number of spatial dimensions is '"
                + std::to_string(SpaceDim) + "'.")
        }

        auto tHostRiNum = Kokkos::create_mirror(tOuput);
        for(Plato::OrdinalType tDim = 0; tDim < SpaceDim; tDim++)
        {
            tHostRiNum(tDim) = tRiNum[tDim];
        }
        Kokkos::deep_copy(tOuput, tHostRiNum);
    }
    else
    {
        Plato::blas1::fill(0.0, tOuput);
    }

    return tOuput;
}
// function dimensionless_richardson_number


/***************************************************************************//**
 * \tparam SpaceDim spatial dimensions (integer)
 *
 * \fn inline Plato::ScalarVector dimensionless_natural_convection_number
 *
 * \brief Parse array with dimensionless natural convection constants (e.g.
 * Rayleigh or Grashof number).
 *
 * \param [in] aInputs input file metadata
 *
 * \return natural convection constants
 ******************************************************************************/
template<Plato::OrdinalType SpaceDim>
inline Plato::ScalarVector
dimensionless_natural_convection_number
(Teuchos::ParameterList & aInputs)
{
    auto tHeatTransfer = Plato::Fluids::heat_transfer_tag(aInputs);
    if( Plato::Fluids::is_dimensionless_parameter_defined("Rayleigh Number", aInputs) &&
            (tHeatTransfer == "natural") )
    {
        return (Plato::Fluids::dimensionless_rayleigh_number<SpaceDim>(aInputs));
    }
    else if( Plato::Fluids::is_dimensionless_parameter_defined("Grashof Number", aInputs) &&
            (tHeatTransfer == "natural" || tHeatTransfer == "mixed") )
    {
        return (Plato::Fluids::dimensionless_grashof_number<SpaceDim>(aInputs));
    }
    else if( Plato::Fluids::is_dimensionless_parameter_defined("Richardson Number", aInputs) &&
            (tHeatTransfer == "mixed") )
    {
        return (Plato::Fluids::dimensionless_richardson_number<SpaceDim>(aInputs));
    }
    else
    {
        THROWERR(std::string("Natural convection properties are not defined. One of these options") +
                 " should be provided: 'Grashof Number' (for natural or mixed convection problems), " +
                 "'Rayleigh Number' (for natural convection problems), or 'Richardson Number' (for mixed convection problems).")
    }
}
// function dimensionless_natural_convection_number


















// todo: predictor equation
/***************************************************************************//**
 * \class VelocityPredictorResidual
 *
 * \tparam PhysicsT    fluid flow physics type
 * \tparam EvaluationT Forward Automatic Differentiation (FAD) evaluation type
 *
 * \brief Evaluate momentum predictor residual, defined as
 *
 * \f[
 *   \mathcal{R}^n_i(w^h_i) = I^n_i(w^h_i) - F^n_i(w^h_i) - S_i^n(w^h_i) - E_i^n(w^h_i) = 0.
 * \f]
 *
 * Inertial Forces:
 *
 * \f[
 *   I^n_i(w^h_i) =
 *     \int_{\Omega}w_i^h\left(\frac{\bar{u}^{\ast}_i - \bar{u}_i^{n}}{\Delta\bar{t}}\right)d\Omega
 * \f]
 *
 * Internal Forces:
 *
 * \f[
 *   F^n_i(w^h_i) =
 *     - \int_{\Omega} w_i^h\left( \bar{u}_j^n\frac{\partial \bar{u}_i^n}{\partial\bar{x}_j} \right) d\Omega
 *     - \int_{\Omega} \frac{\partial w_i^h}{\partial\bar{x}_j}\bar\tau_{ij}^n\,d\Omega
 *     + \int_\Omega w_i^h\left(Gr_i Pr^2\bar{T}^n\right)\,d\Omega
 * \f]
 *
 * Stabilizing Forces:
 *
 * \f[
 *   S_i^n(w^h_i) =
 *     \frac{\Delta\bar{t}}{2}\left[ \int_{\Omega} \frac{\partial w_i^h}{\partial\bar{x}_k}
 *     \left( \bar{u}^n_k \hat{F}^n_{\bar{u}_i} \right) d\Omega \right]
 * \f]
 *
 * where
 *
 * \f[
 *   \hat{F}^n_{\bar{u}_i} = -\bar{u}_j^n \frac{\partial\bar{u}_i^n}{\partial \bar{x}_j} + Gr_i Pr^2\bar{T}^n
 * \f]
 *
 * External Forces:
 *
 * \f[
 *   E_i^n(w^h_i) = \int_{\Gamma-\Gamma_t}w_i^h\bar{\tau}^n_{ij}n_j\,d\Gamma
 * \f]
 *
 ******************************************************************************/
template<typename PhysicsT, typename EvaluationT>
class VelocityPredictorResidual : public Plato::Fluids::AbstractVectorFunction<PhysicsT, EvaluationT>
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
    using ResultT   = typename EvaluationT::ResultScalarType; /*!< result FAD type */
    using ConfigT   = typename EvaluationT::ConfigScalarType; /*!< configuration FAD type */
    using ControlT  = typename EvaluationT::ControlScalarType; /*!< control FAD type */
    using PrevVelT  = typename EvaluationT::PreviousMomentumScalarType; /*!< previous velocity FAD type */
    using PrevTempT = typename EvaluationT::PreviousEnergyScalarType; /*!< previous temperature FAD type */
    using PredVelT  = typename EvaluationT::MomentumPredictorScalarType; /*!< predicted velocity FAD type */

    using AdvectionT  = typename Plato::Fluids::fad_type_t<typename PhysicsT::SimplexT, PrevVelT, ConfigT>; /*!< advection force FAD type */
    using PredStrainT = typename Plato::Fluids::fad_type_t<typename PhysicsT::SimplexT, PredVelT, ConfigT>; /*!< predicted strain rate FAD type */
    using PrevStrainT = typename Plato::Fluids::fad_type_t<typename PhysicsT::SimplexT, PrevVelT, ConfigT>; /*!< previous strain rate FAD type */

    Plato::DataMap& mDataMap; /*!< output database */
    const Plato::SpatialDomain& mSpatialDomain; /*!< Plato spatial model */
    Plato::LinearTetCubRuleDegreeOne<mNumSpatialDims> mCubatureRule; /*!< cubature rule evaluator */

    // set external force evaluators
    std::shared_ptr<Plato::NaturalBCs<mNumSpatialDims, mNumDofsPerNode>> mPrescribedBCs; /*!< prescribed boundary conditions, e.g. tractions */

    // set member scalar data
    Plato::Scalar mTheta = 1.0; /*!< artificial viscous damping */
    Plato::Scalar mBuoyancyConst = 0.0; /*!< dimensionless buoyancy constant */
    Plato::Scalar mViscocity = 1.0; /*!< dimensionless viscocity constant */
    Plato::ScalarVector mNaturalConvectionNum; /*!< dimensionless natural convection number (either Rayleigh or Grashof - depends on user's input) */
    bool mCalculateThermalBuoyancyForces = false; /*!< indicator to determine if thermal buoyancy forces will be considered in calculations */

public:
    /***************************************************************************//**
     * \brief Constructor
     * \param [in] aDomain  holds mesh and entity sets (e.g. node and side sets)
     *   metadata for this spatial domain (e.g. element block)
     * \param [in] aDataMap holds output metadata
     * \param [in] aInputs  input file metadata
     ******************************************************************************/
    VelocityPredictorResidual
    (const Plato::SpatialDomain & aDomain,
     Plato::DataMap             & aDataMap,
     Teuchos::ParameterList     & aInputs) :
         mDataMap(aDataMap),
         mSpatialDomain(aDomain),
         mCubatureRule(Plato::LinearTetCubRuleDegreeOne<mNumSpatialDims>())
    {
        this->setDimensionlessConstants(aInputs);
        this->setAritificalViscousDamping(aInputs);
        this->setNaturalBoundaryConditions(aInputs);
    }

    /***************************************************************************//**
     * \brief Destructor
     ******************************************************************************/
    virtual ~VelocityPredictorResidual(){}

    /***************************************************************************//**
     * \fn void evaluate
     * \brief Evaluate predictor residual.
     * \param [in]  aWorkSets holds state work sets
     * \param [out] aResultWS result work sets
     ******************************************************************************/
    void evaluate
    (const Plato::WorkSets & aWorkSets,
     Plato::ScalarMultiVectorT<ResultT> & aResultWS)
    const override
    {
        auto tNumCells = mSpatialDomain.numCells();
        if( tNumCells != static_cast<Plato::OrdinalType>(aResultWS.extent(0)) )
        {
            THROWERR(std::string("Number of elements mismatch. Spatial domain and output/result workset ")
                + "have different number of cells. " + "Spatial domain has '" + std::to_string(tNumCells)
                + "' elements and output workset has '" + std::to_string(aResultWS.extent(0)) + "' elements.")
        }

        Plato::ScalarVectorT<ConfigT> tCellVolume("cell weight", tNumCells);
        Plato::ScalarArray3DT<ConfigT> tGradient("cell gradient", tNumCells, mNumNodesPerCell, mNumSpatialDims);

        Plato::ScalarArray3DT<PredStrainT> tPredStrainRate("cell strain rate", tNumCells, mNumSpatialDims, mNumSpatialDims);
        Plato::ScalarArray3DT<PrevStrainT> tPrevStrainRate("cell strain rate", tNumCells, mNumSpatialDims, mNumSpatialDims);

        Plato::ScalarMultiVectorT<PrevVelT> tPrevVelGP("previous velocity", tNumCells, mNumSpatialDims);
        Plato::ScalarMultiVectorT<PredVelT> tPredVelGP("predicted velocity", tNumCells, mNumSpatialDims);
        Plato::ScalarMultiVectorT<AdvectionT> tAdvection("advected force", tNumCells, mNumSpatialDims);

        // set local functors
        Plato::ComputeGradientWorkset<mNumSpatialDims> tComputeGradient;
        Plato::InterpolateFromNodal<mNumSpatialDims, mNumVelDofsPerNode, 0/*offset*/, mNumSpatialDims> tIntrplVectorField;

        // set input state worksets
        auto tConfigWS  = Plato::metadata<Plato::ScalarArray3DT<ConfigT>>(aWorkSets.get("configuration"));
        auto tPredVelWS = Plato::metadata<Plato::ScalarMultiVectorT<PredVelT>>(aWorkSets.get("current predictor"));
        auto tPrevVelWS = Plato::metadata<Plato::ScalarMultiVectorT<PrevVelT>>(aWorkSets.get("previous velocity"));
        auto tCriticalTimeStep = Plato::metadata<Plato::ScalarVector>(aWorkSets.get("critical time step"));

        // transfer member data to device
        auto tTheta = mTheta;
        auto tViscocity = mViscocity;

        auto tCubWeight = mCubatureRule.getCubWeight();
        auto tBasisFunctions = mCubatureRule.getBasisFunctions();
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
        {
            tComputeGradient(aCellOrdinal, tGradient, tConfigWS, tCellVolume);
            tCellVolume(aCellOrdinal) *= tCubWeight;

            // 1. add predicted viscous force to residual, i.e. R += \theta K \bar{u}
            Plato::Fluids::strain_rate<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tPredVelWS, tGradient, tPredStrainRate);
            Plato::Fluids::integrate_viscous_forces<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tViscocity, tCellVolume, tGradient, tPredStrainRate, aResultWS, tTheta);

            // 2. add previous viscous force to residual, i.e. R -= (\theta-1)K u_n
            auto tMultiplier = (tTheta - static_cast<Plato::Scalar>(1.0));
            Plato::Fluids::strain_rate<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tPrevVelWS, tGradient, tPrevStrainRate);
            Plato::Fluids::integrate_viscous_forces<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tViscocity, tCellVolume, tGradient, tPrevStrainRate, aResultWS, -tMultiplier);

            // 3. add advection force to residual, i.e. R += C u_n
            tIntrplVectorField(aCellOrdinal, tBasisFunctions, tPrevVelWS, tPrevVelGP);
            Plato::Fluids::calculate_advected_momentum_forces<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tGradient, tPrevVelWS, tPrevVelGP, tAdvection);
            Plato::Fluids::integrate_vector_field<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tBasisFunctions, tCellVolume, tAdvection, aResultWS);

            // 4. apply time step, i.e. \Delta{t}( \theta K\bar{u} + C u_n - (\theta-1)K u_n )
            Plato::blas1::scale<mNumDofsPerCell>(aCellOrdinal, tCriticalTimeStep(0), aResultWS);

            // 5. add predicted inertial force to residual, i.e. R += M\bar{u}
            tIntrplVectorField(aCellOrdinal, tBasisFunctions, tPredVelWS, tPredVelGP);
            Plato::Fluids::integrate_vector_field<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tBasisFunctions, tCellVolume, tPredVelGP, aResultWS);

            // 6. add previous inertial force to residual, i.e. R -= M u_n
            Plato::Fluids::integrate_vector_field<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tBasisFunctions, tCellVolume, tPrevVelGP, aResultWS, -1.0);
        }, "quasi-implicit predicted velocity residual");

        if(mCalculateThermalBuoyancyForces)
        {
            Plato::InterpolateFromNodal<mNumSpatialDims, mNumTempDofsPerNode> tIntrplScalarField;

            // set input and temporary worksets
            Plato::ScalarVectorT<PrevTempT> tPrevTempGP("previous temperature at Gauss point", tNumCells);
            Plato::ScalarMultiVectorT<ResultT>  tThermalBuoyancy("thermal buoyancy", tNumCells, mNumSpatialDims);
            auto tPrevTempWS = Plato::metadata<Plato::ScalarMultiVectorT<PrevTempT>>(aWorkSets.get("previous temperature"));

            // transfer member data to device
            auto tBuoyancyConst = mBuoyancyConst;
            auto tNaturalConvectionNum = mNaturalConvectionNum;
            Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
            {
                // 1. add previous buoyancy force to residual, i.e. R -= (\Delta{t}*Bu*Gr_i) M T_n, where Bu is the buoyancy constant
		auto tMultiplier = static_cast<Plato::Scalar>(-1.0) * tCriticalTimeStep(0);
                tIntrplScalarField(aCellOrdinal, tBasisFunctions, tPrevTempWS, tPrevTempGP);
                Plato::Fluids::calculate_natural_convective_forces<mNumSpatialDims>
                    (aCellOrdinal, tBuoyancyConst, tNaturalConvectionNum, tPrevTempGP, tThermalBuoyancy);
                Plato::Fluids::integrate_vector_field<mNumNodesPerCell, mNumSpatialDims>
                    (aCellOrdinal, tBasisFunctions, tCellVolume, tThermalBuoyancy, aResultWS, tMultiplier);
            }, "add contribution from thermal buoyancy forces to residual");
        }
    }

   /***************************************************************************//**
    * \fn void evaluateBoundary
    * \brief Evaluate non-prescribed boundary forces.
    * \param [in]  aSpatialModel holds mesh and entity sets (e.g. node and side sets) metadata
    * \param [in]  aWorkSets     holds state work sets
    * \param [out] aResultWS     result work sets
    ******************************************************************************/
   void evaluateBoundary
   (const Plato::SpatialModel & aSpatialModel,
    const Plato::WorkSets & aWorkSets,
    Plato::ScalarMultiVectorT<ResultT> & aResult)
   const override
   { return; }

   /***************************************************************************//**
    * \fn void evaluatePrescribed
    * \brief Evaluate prescribed boundary forces.
    * \param [in]  aSpatialModel holds mesh and entity sets (e.g. node and side sets) metadata
    * \param [in]  aWorkSets     holds state work sets
    * \param [out] aResultWS     result work sets
    ******************************************************************************/
   void evaluatePrescribed
   (const Plato::SpatialModel & aSpatialModel,
    const Plato::WorkSets & aWorkSets,
    Plato::ScalarMultiVectorT<ResultT> & aResultWS)
   const override
   {
       if( mPrescribedBCs != nullptr )
       {
           // set input worksets
           auto tControlWS = Plato::metadata<Plato::ScalarMultiVectorT<ControlT>>(aWorkSets.get("control"));
           auto tConfigWS  = Plato::metadata<Plato::ScalarArray3DT<ConfigT>>(aWorkSets.get("configuration"));
           auto tPrevVelWS = Plato::metadata<Plato::ScalarMultiVectorT<PrevVelT>>(aWorkSets.get("previous velocity"));

           // 1. add prescribed traction force to residual
           auto tNumCells = aResultWS.extent(0);
           Plato::ScalarMultiVectorT<ResultT> tTractionWS("traction forces", tNumCells, mNumDofsPerCell);
           mPrescribedBCs->get( aSpatialModel, tPrevVelWS, tControlWS, tConfigWS, tTractionWS);

           // 2. apply time step to traction force
           auto tCriticalTimeStep = Plato::metadata<Plato::ScalarVector>(aWorkSets.get("critical time step"));
           Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
           {
               Plato::blas1::scale<mNumDofsPerCell>(aCellOrdinal, tCriticalTimeStep(0), tTractionWS);
               Plato::blas1::update<mNumDofsPerCell>(aCellOrdinal, -1.0, tTractionWS, 1.0, aResultWS);
           }, "traction force");
       }
   }

private:
   /***************************************************************************//**
    * \fn void setDimensionlessConstants
    * \brief Set dimnesionless constants, e.g. Reynolds number.
    * \param [in] aInputs  input file metadata
    ******************************************************************************/
   void setDimensionlessConstants(Teuchos::ParameterList & aInputs)
   {
       mViscocity = Plato::Fluids::dimensionless_viscosity_constant(aInputs);
       mCalculateThermalBuoyancyForces = Plato::Fluids::calculate_heat_transfer(aInputs);
       if(mCalculateThermalBuoyancyForces)
       {
           mBuoyancyConst = Plato::Fluids::dimensionless_buoyancy_constant(aInputs);
           mNaturalConvectionNum = Plato::Fluids::dimensionless_natural_convection_number<mNumSpatialDims>(aInputs);
       }
   }

   /***************************************************************************//**
    * \fn void setAritificalViscousDamping
    * \brief Set artificial viscous damping, which is a parameter associated to the time integration scheme.
    * \param [in] aInputs  input file metadata
    ******************************************************************************/
   void setAritificalViscousDamping(Teuchos::ParameterList& aInputs)
   {
       if(aInputs.isSublist("Time Integration"))
       {
           auto tTimeIntegration = aInputs.sublist("Time Integration");
           mTheta = tTimeIntegration.get<Plato::Scalar>("Viscosity Damping", 1.0);
       }
   }

   /***************************************************************************//**
    * \fn void setNaturalBoundaryConditions
    * \brief Set natural boundary conditions if defined by the user.
    * \param [in] aInputs  input file metadata
    ******************************************************************************/
   void setNaturalBoundaryConditions(Teuchos::ParameterList& aInputs)
   {
       if(aInputs.isSublist("Momentum Natural Boundary Conditions"))
       {
           auto tInputsNaturalBCs = aInputs.sublist("Momentum Natural Boundary Conditions");
           mPrescribedBCs = std::make_shared<Plato::NaturalBCs<mNumSpatialDims, mNumDofsPerNode>>(tInputsNaturalBCs);
       }
   }
};
// class VelocityPredictorResidual

// todo: predictor equation - to
namespace Brinkman
{

/***************************************************************************//**
 * \class VelocityPredictorResidual
 *
 * \tparam PhysicsT    physics type
 * \tparam EvaluationT forward automatic differentiation evaluation type
 *
 * \brief Calculate the momentum predictor residual, which is defined as
 *
 * \f[
 *   \mathcal{R}^n_i(w^h_i) = I^n_i(w^h_i) - F^n_i(w^h_i) - S_i^n(w^h_i) - E_i^n(w^h_i) = 0.
 * \f]
 *
 * Inertial Forces:
 *
 * \f[
 *   I^n_i(w^h_i) =
 *     \int_{\Omega}w_i^h\left(\frac{\bar{u}^{\ast}_i - \bar{u}_i^{n}}{\Delta\bar{t}}\right)d\Omega
 * \f]
 *
 * Internal Forces:
 *
 * \f[
 *   F^n_i(w^h_i) =
 *     - \int_{\Omega}w_i^h\left( \frac{\partial \bar{u}_j^n}{\partial\bar{x}_j}\bar{u}_i^n
 *     + \bar{u}_j^n\frac{\partial \bar{u}_i^n}{\partial\bar{x}_j} \right) d\Omega
 *     - \int_{\Omega}\frac{\partial w_i^h}{\partial\bar{x}_j}\bar\tau_{ij}^n\,d\Omega
 *     + \int_\Omega w_i^h\left(Gr_i Pr^2\bar{T}^n\right)\,d\Omega
 *     + \int_\Omega w^h_i\left(\pi^{Br}(\theta)\bar{u}_i^n\right)\,d\Omega
 * \f]
 *
 * Stabilizing Forces:
 *
 * \f[
 *   S_i^n(w^h_i) =
 *     \frac{\Delta\bar{t}}{2}\left[ \int_{\Omega} \left( \frac{\partial w_i^h}
 *     {\partial\bar{x}_k}\bar{u}^n_k + w_i^h\frac{\partial \bar{u}^n_k}
 *     {\partial\bar{x}_k} \right) \hat{F}^n_{\bar{u}_i}\, d\Omega \right]
 * \f]
 *
 * where
 *
 * \f[
 *   \hat{F}^n_{\bar{u}_i} =
 *     -\frac{\partial\bar{u}_j^n}{\partial \bar{x}_j}\bar{u}_i^n
 *     - \bar{u}_j^n\frac{\partial\bar{u}_i^n}{\partial \bar{x}_j}
 *     + Gr_i Pr^2\bar{T}^n + \pi^{Br}(\theta)\bar{u}_i^n
 * \f]
 *
 * External Forces:
 *
 * \f[
 *   E_i^n(w^h_i) =
 *     \int_{\Gamma-\Gamma_t}w_i^h\bar{\tau}^n_{ij}n_j\,d\Gamma
 *     + \int_{\Gamma_t}w_i^h\left( t_i+\bar{p}^{n}n_i \right)\,d\Gamma
 * \f]
 *
 ******************************************************************************/
template<typename PhysicsT, typename EvaluationT>
class VelocityPredictorResidual : public Plato::Fluids::AbstractVectorFunction<PhysicsT, EvaluationT>
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
    using ResultT   = typename EvaluationT::ResultScalarType;
    using ConfigT   = typename EvaluationT::ConfigScalarType;
    using ControlT  = typename EvaluationT::ControlScalarType;
    using PrevVelT  = typename EvaluationT::PreviousMomentumScalarType;
    using PrevTempT = typename EvaluationT::PreviousEnergyScalarType;
    using PredVelT  = typename EvaluationT::MomentumPredictorScalarType;
    using StrainT   = typename Plato::Fluids::fad_type_t<typename PhysicsT::SimplexT, PrevVelT, ConfigT>;

    Plato::DataMap& mDataMap; /*!< output database */
    const Plato::SpatialDomain& mSpatialDomain; /*!< Plato spatial model */
    Plato::LinearTetCubRuleDegreeOne<mNumSpatialDims> mCubatureRule; /*!< cubature rule evaluator */
    std::shared_ptr<Plato::NaturalBCs<mNumSpatialDims, mNumDofsPerNode>> mPrescribedBCs; /*!< prescribed boundary conditions, e.g. tractions */

    // set member scalar data
    Plato::Scalar mBuoyancy = 1.0; /*!< buoyancy dimensionless number */
    Plato::Scalar mViscocity = 1.0; /*!< viscocity dimensionless number */
    Plato::Scalar mPermeability = 1.0; /*!< permeability dimensionless number */
    Plato::Scalar mBrinkmanConvexityParam = 0.5;  /*!< brinkman model convexity parameter */
    Plato::ScalarVector mGrNum; /*!< grashof dimensionless number */

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
    }

    virtual ~VelocityPredictorResidual(){}

    /***************************************************************************//**
     * \fn void evaluate
     * \brief Evaluate the total internal forces, which are given by the sum of the
     *   inertial, internal, and stabilizing forces. The penalized internal and
     *   stabilizing forces for density-based topology optimization are respectively
     *   calculated as:
     *
     * Inertial:
     *
     * \f[
     *   I^n_i(w^h_i) =
     *     \int_{\Omega}w_i^h\left(\frac{\bar{u}^{\ast}_i - \bar{u}_i^{n}}{\Delta\bar{t}}\right)d\Omega
     * \f]
     *
     * Internal:
     *
     * \f[
     *   F^n_i(w^h_i) =
     *     - \int_{\Omega}w_i^h\left( \frac{\partial \bar{u}_j^n}{\partial\bar{x}_j}\bar{u}_i^n
     *     + \bar{u}_j^n\frac{\partial \bar{u}_i^n}{\partial\bar{x}_j} \right) d\Omega
     *     - \int_{\Omega}\frac{\partial w_i^h}{\partial\bar{x}_j}\bar\tau_{ij}^n\,d\Omega
     *     + \int_\Omega w_i^h\left(Gr_i Pr^2\bar{T}^n\right)\,d\Omega
     *     + \int_\Omega w^h_i\left(\pi^{Br}(\theta)\bar{u}_i^n\right)\,d\Omega
     * \f]
     *
     * Stabilizing:
     *
     * \f[
     *   S_i^n(w^h_i) =
     *     \frac{\Delta\bar{t}}{2}\left[ \int_{\Omega} \left( \frac{\partial w_i^h}
     *     {\partial\bar{x}_k}\bar{u}^n_k + w_i^h\frac{\partial \bar{u}^n_k}
     *     {\partial\bar{x}_k} \right) \hat{F}^n_{\bar{u}_i}\, d\Omega \right]
     * \f]
     *
     * where
     *
     * \f[
     *   \hat{F}^n_{\bar{u}_i} =
     *     -\frac{\partial\bar{u}_j^n}{\partial \bar{x}_j}\bar{u}_i^n
     *     - \bar{u}_j^n\frac{\partial\bar{u}_i^n}{\partial \bar{x}_j}
     *     + Gr_i Pr^2\bar{T}^n + \pi^{Br}(\theta)\bar{u}_i^n
     * \f]
     *
     * Finally, the internal momentum corrector residual is given by
     *
     * \f[
     *   \hat{R}_i^n(w^h_i) = I^n_i(w^h_i) - F^n_i(w^h_i) - S_i^n(w^h_i) = 0.
     * \f]
     *
     * In the equations presented above \f$ \alpha \f$ denotes a scalar multiplier,
     * \f$ w_i^h \f$ are the test functions, \f$ \Delta\bar{t} \f$ denotes the current
     * time step, \f$ \bar{\tau}^n_{ij} \f$ is the second order deviatoric stress tensor,
     * \f$ \theta \f$ denotes the physical design variables (i.e. density field), and
     * \f$ n_i \f$ is the unit normal vector.
     *
     ******************************************************************************/
    void evaluate
    (const Plato::WorkSets & aWorkSets,
     Plato::ScalarMultiVectorT<ResultT> & aResultWS)
    const override
    {
        auto tNumCells = mSpatialDomain.numCells();
        if( tNumCells != static_cast<Plato::OrdinalType>(aResultWS.extent(0)) )
        {
            THROWERR(std::string("Number of elements mismatch. Spatial domain and output/result workset ")
                + "have different number of cells. " + "Spatial domain has '" + std::to_string(tNumCells)
                + "' elements and output workset has '" + std::to_string(aResultWS.extent(0)) + "' elements.")
        }

        Plato::ScalarVectorT<ConfigT>   tCellVolume("cell weight", tNumCells);
        Plato::ScalarVectorT<PrevTempT> tPrevTempGP("previous temperature at Gauss point", tNumCells);

        Plato::ScalarArray3DT<ConfigT> tGradient("cell gradient", tNumCells, mNumNodesPerCell, mNumSpatialDims);
        Plato::ScalarArray3DT<StrainT> tStrainRate("cell strain rate", tNumCells, mNumSpatialDims, mNumSpatialDims);

        Plato::ScalarMultiVectorT<PrevVelT> tPrevVelGP("previous velocity", tNumCells, mNumSpatialDims);
        Plato::ScalarMultiVectorT<PredVelT> tPredVelGP("predicted velocity", tNumCells, mNumSpatialDims);

        Plato::ScalarMultiVectorT<ResultT>  tBrinkman("brinkman", tNumCells, mNumSpatialDims);
        Plato::ScalarMultiVectorT<ResultT>  tAdvection("advection", tNumCells, mNumSpatialDims);
        Plato::ScalarMultiVectorT<ResultT>  tStabilization("stabilization", tNumCells, mNumSpatialDims);
        Plato::ScalarMultiVectorT<ResultT>  tNaturalConvection("natural convection", tNumCells, mNumSpatialDims);

        Plato::ScalarMultiVectorT<ResultT>  tStabForces("stabilizing forces", tNumCells, mNumDofsPerCell);
        Plato::ScalarMultiVectorT<ResultT>  tInternalForces("internal forces", tNumCells, mNumDofsPerCell);

        // set local functors
        Plato::ComputeGradientWorkset<mNumSpatialDims> tComputeGradient;
        Plato::InterpolateFromNodal<mNumSpatialDims, mNumTempDofsPerNode> tIntrplScalarField;
        Plato::InterpolateFromNodal<mNumSpatialDims, mNumVelDofsPerNode, 0/*offset*/, mNumSpatialDims> tIntrplVectorField;

        // set input state worksets
        auto tControlWS   = Plato::metadata<Plato::ScalarMultiVectorT<ControlT>>(aWorkSets.get("control"));
        auto tConfigWS    = Plato::metadata<Plato::ScalarArray3DT<ConfigT>>(aWorkSets.get("configuration"));
        auto tPrevVelWS   = Plato::metadata<Plato::ScalarMultiVectorT<PrevVelT>>(aWorkSets.get("previous velocity"));
        auto tPrevTempWS  = Plato::metadata<Plato::ScalarMultiVectorT<PrevTempT>>(aWorkSets.get("previous temperature"));
        auto tPredictorWS = Plato::metadata<Plato::ScalarMultiVectorT<PredVelT>>(aWorkSets.get("current predictor"));
        auto tCriticalTimeStep = Plato::metadata<Plato::ScalarVector>(aWorkSets.get("critical time step"));

        // transfer member data to device
        auto tGrNum = mGrNum;
        auto tBuoyancy = mBuoyancy;
        auto tViscocity = mViscocity;
        auto tPermeability = mPermeability;
        auto tBrinkmanConvexityParam = mBrinkmanConvexityParam;

        auto tCubWeight = mCubatureRule.getCubWeight();
        auto tBasisFunctions = mCubatureRule.getBasisFunctions();
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
        {
            tComputeGradient(aCellOrdinal, tGradient, tConfigWS, tCellVolume);
            tCellVolume(aCellOrdinal) *= tCubWeight;

            // 1. calculate internal force contribution
            Plato::Fluids::strain_rate<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tPrevVelWS, tGradient, tStrainRate);
            Plato::Fluids::integrate_viscous_forces<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tViscocity, tCellVolume, tGradient, tStrainRate, tInternalForces, -1.0);

            tIntrplVectorField(aCellOrdinal, tBasisFunctions, tPrevVelWS, tPrevVelGP);
            Plato::Fluids::calculate_advected_momentum_forces<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tGradient, tPrevVelWS, tPrevVelGP, tAdvection);
            Plato::Fluids::integrate_vector_field<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tBasisFunctions, tCellVolume, tAdvection, tInternalForces, -1.0);

            tIntrplScalarField(aCellOrdinal, tBasisFunctions, tPrevTempWS, tPrevTempGP);
            Plato::Fluids::calculate_natural_convective_forces<mNumSpatialDims>
                (aCellOrdinal, tBuoyancy, tGrNum, tPrevTempGP, tNaturalConvection);
            Plato::Fluids::integrate_vector_field<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tBasisFunctions, tCellVolume, tNaturalConvection, tInternalForces);

            ControlT tPenalizedPermeability = Plato::Fluids::brinkman_penalization<mNumNodesPerCell>
                (aCellOrdinal, tPermeability, tBrinkmanConvexityParam, tControlWS);
            Plato::Fluids::calculate_brinkman_forces<mNumSpatialDims>
                (aCellOrdinal, tPenalizedPermeability, tPrevVelGP, tBrinkman);
            Plato::Fluids::integrate_vector_field<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tBasisFunctions, tCellVolume, tBrinkman, tInternalForces);
            Plato::blas1::scale<mNumDofsPerCell>(aCellOrdinal, tCriticalTimeStep(0), tStabilization);

            // 2. calculate stabilization term
            Plato::blas1::update<mNumSpatialDims>(aCellOrdinal, -1.0, tAdvection, 1.0, tStabilization);
            Plato::blas1::update<mNumSpatialDims>(aCellOrdinal,  1.0, tBrinkman , 1.0, tStabilization);
            Plato::blas1::update<mNumSpatialDims>(aCellOrdinal,  1.0, tNaturalConvection, 1.0, tStabilization);
            Plato::Fluids::integrate_stabilizing_vector_force<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tCellVolume, tGradient, tPrevVelGP, tStabilization, tStabForces);
            auto tMultiplier = static_cast<Plato::Scalar>(0.5) * tCriticalTimeStep(0) * tCriticalTimeStep(0);
            Plato::blas1::scale<mNumDofsPerCell>(aCellOrdinal, tMultiplier, tStabilization);

            // 3. calculate residual
            tIntrplVectorField(aCellOrdinal, tBasisFunctions, tPredictorWS, tPredVelGP);
            Plato::Fluids::integrate_momentum_inertial_forces<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tBasisFunctions, tCellVolume, tPredVelGP, tPrevVelGP, aResultWS);
            Plato::blas1::update<mNumDofsPerCell>(aCellOrdinal, -1.0, tStabForces, 1.0, aResultWS);
            Plato::blas1::update<mNumDofsPerCell>(aCellOrdinal, -1.0, tInternalForces, 1.0, aResultWS);
        }, "predictor residual");
    }

    /***************************************************************************//**
     * \fn void evaluateBoundary
     * \brief Evaluate deviatoric traction forces on non-traction boundary, which
     * are defined as
     *
     * \f[
     *   \alpha\Delta\bar{t}\int_{\Gamma-\Gamma_t}w_i^h\bar{\tau}^n_{ij}n_j\,d\Gamma
     * \f]
     *
     * where \f$ \alpha \f$ denotes a scalar multiplier, \f$ w_i^h \f$ are the
     * test functions, \f$ \Delta\bar{t} \f$ denotes the current time step,
     * \f$ \bar{\tau}^n_{ij} \f$ is the second order deviatoric stress tensor,
     * and \f$ n_i \f$ is the unit normal vector.
     *
     ******************************************************************************/
   void evaluateBoundary
   (const Plato::SpatialModel & aSpatialModel,
    const Plato::WorkSets & aWorkSets,
    Plato::ScalarMultiVectorT<ResultT> & aResult)
   const override
   { return; }

   /***************************************************************************//**
    * \fn void evaluatePrescribed
    * \brief Evaluate prescribed deviatoric traction forces, which are defined as
    *
    * \f[
    *   \alpha\Delta\bar{t}\int_{\Gamma_t}w_i^h\left( t_i+\bar{p}^{n}n_i \right)\,d\Gamma
    * \f]
    *
    * where \f$\alpha\f$ denotes a scalar multiplier, \f$\Delta\bar{t}\f$ denotes
    * the current time step, \f${t}_i\f$ is the i-th component of the prescribed
    * traction force, \f${p}^{n}\f$ is the previous pressure, and \f${n}_{i}\f$
    * is the unit normal vector.
    *
    ******************************************************************************/
   void evaluatePrescribed
   (const Plato::SpatialModel & aSpatialModel,
    const Plato::WorkSets & aWorkSets,
    Plato::ScalarMultiVectorT<ResultT> & aResultWS)
   const override
   {
       if( mPrescribedBCs != nullptr )
       {
           // set input worksets
           auto tControlWS = Plato::metadata<Plato::ScalarMultiVectorT<ControlT>>(aWorkSets.get("control"));
           auto tConfigWS  = Plato::metadata<Plato::ScalarArray3DT<ConfigT>>(aWorkSets.get("configuration"));
           auto tPrevVelWS = Plato::metadata<Plato::ScalarMultiVectorT<PrevVelT>>(aWorkSets.get("previous velocity"));

           // 1. add prescribed deviatoric traction forces
           auto tNumCells = aResultWS.extent(0);
           Plato::ScalarMultiVectorT<ResultT> tResultWS("traction forces", tNumCells, mNumDofsPerCell);
           mPrescribedBCs->get( aSpatialModel, tPrevVelWS, tControlWS, tConfigWS, tResultWS); // prescribed traction forces

           // 3. multiply force vector by the corresponding nodal time steps
           auto tCriticalTimeStep = Plato::metadata<Plato::ScalarVector>(aWorkSets.get("critical time step"));
           Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
           {
               Plato::blas1::scale<mNumDofsPerCell>(aCellOrdinal, tCriticalTimeStep(0), aResultWS);
               Plato::blas1::update<mNumDofsPerCell>(aCellOrdinal, -1.0, tResultWS, 1.0, aResultWS);
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
       this->setDimensionlessProperties(aInputs);
       auto tHyperbolic = aInputs.sublist("Hyperbolic");
       if(tHyperbolic.isSublist("Momentum Conservation"))
       {
           auto tMomentumParamList = tHyperbolic.sublist("Momentum Conservation");
           this->setPenaltyModel(tMomentumParamList);
       }
   }

   void setPenaltyModel
   (Teuchos::ParameterList & aInputs)
   {
        if (aInputs.isSublist("Penalty Function"))
        {
            auto tPenaltyFuncList = aInputs.sublist("Penalty Function");
            mBrinkmanConvexityParam = tPenaltyFuncList.get<Plato::Scalar>("Brinkman Convexity Parameter", 0.5);
        }
   }

   void setPermeability
   (Teuchos::ParameterList & aInputs)
   {
       auto tHyperbolic = aInputs.sublist("Hyperbolic");
       auto tDaNum = Plato::parse_parameter<Plato::Scalar>("Darcy Number", "Dimensionless Properties", tHyperbolic);
       auto tPrNum = Plato::parse_parameter<Plato::Scalar>("Prandtl Number", "Dimensionless Properties", tHyperbolic);
       mPermeability = tPrNum / tDaNum;
   }

   void setViscosity
   (Teuchos::ParameterList & aInputs)
   {
       auto tHyperbolic = aInputs.sublist("Hyperbolic");
       auto tTag = tHyperbolic.get<std::string>("Heat Transfer", "None");
       auto tHeatTransfer = Plato::tolower(tTag);

       if(tHeatTransfer == "forced" || tHeatTransfer == "none")
       {
           auto tReNum = Plato::parse_parameter<Plato::Scalar>("Reynolds Number", "Dimensionless Properties", tHyperbolic);
           mViscocity = static_cast<Plato::Scalar>(1) / tReNum;
       }
       else if(tHeatTransfer == "natural")
       {
           mViscocity = Plato::parse_parameter<Plato::Scalar>("Prandtl Number", "Dimensionless Properties", tHyperbolic);
       }
       else
       {
           THROWERR(std::string("'Heat Transfer' mechanism with tag '") + tHeatTransfer + "' is not supported.")
       }
   }

   void setGrashofNumber
   (Teuchos::ParameterList & aInputs)
   {
        auto tHyperbolic = aInputs.sublist("Hyperbolic");
        auto tTag = tHyperbolic.get<std::string>("Heat Transfer", "None");
        auto tHeatTransfer = Plato::tolower(tTag);
        auto tCalculateHeatTransfer = tHeatTransfer == "none" ? false : true;

        if(tCalculateHeatTransfer)
        {
            auto tGrNum = Plato::parse_parameter<Teuchos::Array<Plato::Scalar>>("Grashof Number", "Dimensionless Properties", tHyperbolic);
            if(tGrNum.size() != mNumSpatialDims)
            {
                THROWERR(std::string("'Grashof Number' array length should match the number of physical spatial dimensions. ")
                    + "Array length is '" + std::to_string(tGrNum.size()) + "' and the number of physical spatial dimensions is '"
                    + std::to_string(mNumSpatialDims) + "'.")
            }

            auto tLength = mGrNum.size();
            auto tHostGrNum = Kokkos::create_mirror(mGrNum);
            for(decltype(tLength) tDim = 0; tDim < tLength; tDim++)
            {
                tHostGrNum(tDim) = tGrNum[tDim];
            }
            Kokkos::deep_copy(mGrNum, tHostGrNum);
        }
        else
        {
            Plato::blas1::fill(0.0, mGrNum);
        }
   }

   void setBuoyancyConstant
   (Teuchos::ParameterList & aInputs)
   {
       auto tHyperbolic = aInputs.sublist("Hyperbolic");
       auto tTag = tHyperbolic.get<std::string>("Heat Transfer", "None");
       auto tHeatTransfer = Plato::tolower(tTag);

       if(tHeatTransfer == "forced")
       {
           auto tReNum = Plato::parse_parameter<Plato::Scalar>("Reynolds Number", "Dimensionless Properties", tHyperbolic);
           mBuoyancy = static_cast<Plato::Scalar>(1) / (tReNum*tReNum);
       }
       else if(tHeatTransfer == "natural")
       {
           auto tPrNum = Plato::parse_parameter<Plato::Scalar>("Prandtl Number", "Dimensionless Properties", tHyperbolic);
           mBuoyancy = tPrNum*tPrNum;
       }
       else if(tHeatTransfer == "none")
       {
           mBuoyancy = 0;
       }
       else
       {
           THROWERR(std::string("'Heat Transfer' mechanism with tag '") + tHeatTransfer + "' is not supported.")
       }
   }

   void setDimensionlessProperties
   (Teuchos::ParameterList & aInputs)
   {
       if(aInputs.isSublist("Hyperbolic") == false)
       {
           THROWERR("'Hyperbolic' Parameter List is not defined.")
       }
       this->setViscosity(aInputs);
       this->setPermeability(aInputs);
       this->setGrashofNumber(aInputs);
       this->setBuoyancyConstant(aInputs);
   }

    void setNaturalBoundaryConditions(Teuchos::ParameterList& aInputs)
    {
        if(aInputs.isSublist("Momentum Natural Boundary Conditions"))
        {
            auto tInputsNaturalBCs = aInputs.sublist("Momentum Natural Boundary Conditions");
            mPrescribedBCs = std::make_shared<Plato::NaturalBCs<mNumSpatialDims, mNumDofsPerNode>>(tInputsNaturalBCs);
        }
    }
};
// class VelocityPredictorResidual

}
// namespace Brinkman

/***************************************************************************//**
 * \fn device_type void calculate_pressure_gradient
 * \brief Calculate pressure gradient, which is defined as
 *
 * \f[
 *   \frac{\partial p^{n+\theta_2}}{\partial x_i} =
 *     \alpha\left( 1-\theta_2 \right)\frac{partial p^n}{partial x_i}
 *     + \theta_2\frac{\partial\delta{p}}{\partial x_i}
 * \f]
 *
 * where \f$ \delta{p} = p^{n+1} - p^{n} \f$ and \f$ \alpha \f$ denotes a scalar
 * multiplier.
 *
 ******************************************************************************/
template<Plato::OrdinalType NumNodes,
         Plato::OrdinalType SpaceDim,
         typename ConfigT,
         typename CurPressT,
         typename PrevPressT,
         typename PressGradT>
DEVICE_TYPE inline void
calculate_pressure_gradient
(const Plato::OrdinalType & aCellOrdinal,
 const Plato::Scalar & aTheta,
 const Plato::ScalarArray3DT<ConfigT> & aGradient,
 const Plato::ScalarMultiVectorT<CurPressT> & aCurPress,
 const Plato::ScalarMultiVectorT<PrevPressT> & aPrevPress,
 const Plato::ScalarMultiVectorT<PressGradT> & aPressGrad)
{
    for(Plato::OrdinalType tNode = 0; tNode < NumNodes; tNode++)
    {
        for(Plato::OrdinalType tDim = 0; tDim < SpaceDim; tDim++)
        {
            aPressGrad(aCellOrdinal, tDim) += ( (static_cast<Plato::Scalar>(1.0) - aTheta)
                * aGradient(aCellOrdinal, tNode, tDim) * aPrevPress(aCellOrdinal, tNode) )
                + ( aTheta * aGradient(aCellOrdinal, tNode, tDim) * aCurPress(aCellOrdinal, tNode) );
        }
    }
}

template<Plato::OrdinalType NumNodes,
         Plato::OrdinalType SpaceDim,
         typename ConfigT,
         typename FieldT,
         typename FieldGradT>
DEVICE_TYPE inline void
calculate_scalar_field_gradient
(const Plato::OrdinalType & aCellOrdinal,
 const Plato::ScalarArray3DT<ConfigT> & aGradient,
 const Plato::ScalarMultiVectorT<FieldT> & aScalarField,
 const Plato::ScalarMultiVectorT<FieldGradT> & aResult)
{
    for(Plato::OrdinalType tNode = 0; tNode < NumNodes; tNode++)
    {
        for(Plato::OrdinalType tDim = 0; tDim < SpaceDim; tDim++)
        {
            aResult(aCellOrdinal, tDim) += aGradient(aCellOrdinal, tNode, tDim) * aScalarField(aCellOrdinal, tNode);
        }
    }
}

/***************************************************************************//**
 * \fn device_type void integrate_stabilizing_pressure_gradient
 * \brief Integrate stabilizing pressure gradient, which is given by
 *
 * \f[
 *   \alpha\int_{\Omega} \left( \frac{\partial w_i^h}{\partial\bar{x}_k}\bar{u}^n_k \right)
 *     \frac{\partial \bar{p}^{n+\theta_2}}{\partial \bar{x}_i}\, d\Omega
 * \f]
 *
 * where \f$ \frac{\partial \bar{p}^{n+\theta_2}}{\partial \bar{x}_i} \f$ is given by
 *
 * \f[
 *   \frac{\partial \bar{p}^{n+\theta_2}}{\partial \bar{x}_i} =
 *     \alpha\left( 1-\theta_2 \right)\frac{partial \bar{p}^n}{partial \bar{x}_i}
 *       + \theta_2\frac{\partial\delta\bar{p}}{\partial \bar{x}_i}
 * \f]
 *
 ******************************************************************************/
template
<Plato::OrdinalType NumNodes,
 Plato::OrdinalType SpaceDim,
 typename ResultT,
 typename ConfigT,
 typename PrevVelT,
 typename StabilityT>
DEVICE_TYPE inline void
integrate_stabilizing_pressure_gradient
(const Plato::OrdinalType & aCellOrdinal,
 const Plato::ScalarVectorT<ConfigT> & aCellVolume,
 const Plato::ScalarArray3DT<ConfigT> & aGradient,
 const Plato::ScalarMultiVectorT<PrevVelT> & aPrevVelGP,
 const Plato::ScalarMultiVectorT<StabilityT> & aPressGrad,
 const Plato::ScalarMultiVectorT<ResultT> & aResult,
 Plato::Scalar aMultiplier = 1.0)
{
    for(Plato::OrdinalType tNode = 0; tNode < NumNodes; tNode++)
    {
        for(Plato::OrdinalType tDimI = 0; tDimI < SpaceDim; tDimI++)
        {
            auto tLocalCellDof = (SpaceDim * tNode) + tDimI;
            for(Plato::OrdinalType tDimK = 0; tDimK < SpaceDim; tDimK++)
            {
                aResult(aCellOrdinal, tLocalCellDof) += aMultiplier * ( ( aGradient(aCellOrdinal, tNode, tDimK) *
                    aPrevVelGP(aCellOrdinal, tDimK) ) * aPressGrad(aCellOrdinal, tDimI) ) * aCellVolume(aCellOrdinal);
            }
        }
    }
}

// todo: corrector equation
/***************************************************************************//**
 * \class VelocityCorrectorResidual
 *
 * \tparam PhysicsT    physics type
 * \tparam EvaluationT forward automatic differentiation evaluation type
 *
 * \brief Evaluate momentum corrector residual, which is defined by
 *
 * \f[
 *   \mathcal{R}_i^n(w^h_i) = I^n_i(w^h_i) - F^n_i(w^h_i) - S_i^n(w^h_i) - E_i^n(w^h_i) = 0.
 * \f]
 *
 * where
 *
 * Inertial:
 *
 * \f[
 *   I^n_i(w^h_i) = \int_{\Omega}w_i^h \left(\frac{\bar{u}^{n+1}_i - \bar{u}^{\ast}_i}{\Delta\bar{t}}\right)d\Omega
 * \f]
 *
 * Internal:
 *
 * \f[
 *   F^n_i(w^h_i) = -\int_{\Omega}w_i^h\frac{\partial\bar{p}^{\,n+\theta_2}}{\partial\bar{x}_i}d\Omega
 * \f]
 *
 * Stabilizing:
 *
 * \f[
 *   S^n_i(w^h_i) =
 *     -\frac{\Delta\bar{t}}{2}\int_{\Omega} \left(  \frac{\partial w_i^h}{\partial\bar{x}_k}\bar{u}^{n}_k
 *     + w_i^h \frac{\partial\bar{u}^{n}_k}{\partial\bar{x}_k} \right)
 *       \frac{\partial\bar{p}^{\,n+\theta_2}}{\partial\bar{x}_i} d\Omega
 * \f]
 *
 * where
 *
 * \f[
 *   \frac{\partial p^{n+\theta_2}}{\partial x_i} =
 *     \left( 1-\theta_2 \right)\frac{partial p^n}{partial x_i}
 *       + \theta_2\frac{\partial\delta{p}}{\partial x_i}
 * \f]
 *
 * The external forces \f$ E_i^n(w^h_i) \f$ are zero.
 *
 ******************************************************************************/
template<typename PhysicsT, typename EvaluationT>
class VelocityCorrectorResidual : public Plato::Fluids::AbstractVectorFunction<PhysicsT, EvaluationT>
{
private:
    static constexpr auto mNumDofsPerNode    = PhysicsT::mNumDofsPerNode; /*!< number of degrees of freedom per node */
    static constexpr auto mNumDofsPerCell    = PhysicsT::mNumDofsPerCell; /*!< number of degrees of freedom per cell */
    static constexpr auto mNumNodesPerCell   = PhysicsT::SimplexT::mNumNodesPerCell; /*!< number of nodes per cell */
    static constexpr auto mNumSpatialDims    = PhysicsT::SimplexT::mNumSpatialDims;         /*!< number of spatial dimensions */

    using ResultT    = typename EvaluationT::ResultScalarType;
    using ConfigT    = typename EvaluationT::ConfigScalarType;
    using CurVelT    = typename EvaluationT::CurrentMomentumScalarType;
    using PrevVelT   = typename EvaluationT::PreviousMomentumScalarType;
    using PredVelT   = typename EvaluationT::MomentumPredictorScalarType;
    using CurPressT  = typename EvaluationT::CurrentMassScalarType;
    using PrevPressT = typename EvaluationT::PreviousMassScalarType;

    //using CurrentStrainT = typename Plato::Fluids::fad_type_t<typename PhysicsT::SimplexT, CurVelT, ConfigT>;
    //using PrevPressGradT = typename Plato::Fluids::fad_type_t<typename PhysicsT::SimplexT, PrevPressT, ConfigT>;
    using PressGradT = typename Plato::Fluids::fad_type_t<typename PhysicsT::SimplexT, CurPressT, PrevPressT, ConfigT>;

    Plato::DataMap& mDataMap;                   /*!< output database */
    const Plato::SpatialDomain& mSpatialDomain; /*!< Plato spatial model */
    Plato::LinearTetCubRuleDegreeOne<mNumSpatialDims> mCubatureRule; /*!< integration rule */

    //Plato::Scalar mViscocity = 1.0; /*!< dimensionless viscocity constant */
    Plato::Scalar mPressureTheta = 1.0; /*!< artificial pressure damping */
    Plato::Scalar mViscosityTheta = 1.0; /*!< artificial viscosity damping */

public:
    VelocityCorrectorResidual
    (const Plato::SpatialDomain & aDomain,
     Plato::DataMap             & aDataMap,
     Teuchos::ParameterList     & aInputs) :
         mDataMap(aDataMap),
         mSpatialDomain(aDomain),
         mCubatureRule(Plato::LinearTetCubRuleDegreeOne<mNumSpatialDims>())
    {
        this->setAritificalPressureDamping(aInputs);
        //mViscocity = Plato::Fluids::dimensionless_viscosity_constant(aInputs);
    }

    virtual ~VelocityCorrectorResidual(){}

    /***************************************************************************//**
     * \fn void evaluate
     * \brief Evaluate the internal momentum corrector residual, which is defined by
     * the sum of the inertial, internal, and stabilizing forces, defined as
     *
     * Inertial:
     *
     * \f[
     *   I^n_i(w^h_i) = \int_{\Omega}w_i^h \left(\frac{\bar{u}^{n+1}_i - \bar{u}^{\ast}_i}{\Delta\bar{t}}\right)d\Omega
     * \f]
     *
     * Internal:
     *
     * \f[
     *   F^n_i(w^h_i) = -\int_{\Omega}w_i^h\frac{\partial\bar{p}^{\,n+\theta_2}}{\partial\bar{x}_i}d\Omega
     * \f]
     *
     * Stabilizing:
     *
     * \f[
     *   S^n_i(w^h_i) =
     *     -\frac{\Delta\bar{t}}{2}\int_{\Omega} \left(  \frac{\partial w_i^h}{\partial\bar{x}_k}\bar{u}^{n}_k
     *     + w_i^h \frac{\partial\bar{u}^{n}_k}{\partial\bar{x}_k} \right)
     *       \frac{\partial\bar{p}^{\,n+\theta_2}}{\partial\bar{x}_i} d\Omega
     * \f]
     *
     * where
     *
     * \f[
     *   \frac{\partial p^{n+\theta_2}}{\partial x_i} =
     *     \left( 1-\theta_2 \right)\frac{partial p^n}{partial x_i} + \theta_2\frac{\partial\delta{p}}{\partial x_i}
     * \f]
     *
     * Finally, the internal momentum corrector residual is given by
     *
     * \f[
     *   \hat{R}_i^n(w^h_i) = I^n_i(w^h_i) - F^n_i(w^h_i) - S_i^n(w^h_i) = 0.
     * \f]
     *
     * In the equations presented above \f$ w_i^h \f$ denote the test functions, \f$ \Delta\bar{t} \f$
     * denotes the current time step, \f$ \bar{u}^{n+1} \f$ and \f$ \bar{u}^n \f$ are respectively the
     * current and previous velocity \f$ \bar{u}^{\ast}_i \f$ is the predicted velocity (i.e. predictor),
     * \f$ \bar{p}^{n+1} \f$ and \f$ \bar{u}^n \f$ are respectively the current and previous pressure,
     * \f$ \theta_2 \f$ is a scalar multiplier, and \f$ \delta{p} = p^{n+1} - p^{n}. \f$
     *
     ******************************************************************************/
    void evaluate
    (const Plato::WorkSets & aWorkSets,
     Plato::ScalarMultiVectorT<ResultT> & aResultWS)
    const override
    {
        auto tNumCells = mSpatialDomain.numCells();
        if( tNumCells != static_cast<Plato::OrdinalType>(aResultWS.extent(0)) )
        {
            THROWERR(std::string("Number of elements mismatch. Spatial domain and output/result workset ")
                + "have different number of cells. " + "Spatial domain has '" + std::to_string(tNumCells)
                + "' elements and output workset has '" + std::to_string(aResultWS.extent(0)) + "' elements.")
        }

        // set local data structures
        Plato::ScalarVectorT<ConfigT> tCellVolume("cell weight", tNumCells);
        Plato::ScalarArray3DT<ConfigT> tGradient("cell gradient", tNumCells, mNumNodesPerCell, mNumSpatialDims);
        //Plato::ScalarArray3DT<CurrentStrainT> tCurStrainRate("cell strain rate", tNumCells, mNumSpatialDims, mNumSpatialDims);

        Plato::ScalarMultiVectorT<PressGradT> tPressGradGP("pressure gradient", tNumCells, mNumSpatialDims);
        Plato::ScalarMultiVectorT<CurVelT> tCurVelGP("current velocity at Gauss points", tNumCells, mNumSpatialDims);
        Plato::ScalarMultiVectorT<PrevVelT> tPrevVelGP("previous velocity at Gauss points", tNumCells, mNumSpatialDims);
        Plato::ScalarMultiVectorT<PredVelT> tPredVelGP("predicted velocity at Gauss points", tNumCells, mNumSpatialDims);
        //Plato::ScalarMultiVectorT<PrevPressGradT> tPrevPressGradGP("previous pressure gradient", tNumCells, mNumSpatialDims);

        // set input state worksets
        auto tConfigWS    = Plato::metadata<Plato::ScalarArray3DT<ConfigT>>(aWorkSets.get("configuration"));
        auto tCurVelWS    = Plato::metadata<Plato::ScalarMultiVectorT<CurVelT>>(aWorkSets.get("current velocity"));
        auto tPrevVelWS   = Plato::metadata<Plato::ScalarMultiVectorT<PrevVelT>>(aWorkSets.get("previous velocity"));
        auto tPredVelWS   = Plato::metadata<Plato::ScalarMultiVectorT<PredVelT>>(aWorkSets.get("current predictor"));
        auto tCurPressWS  = Plato::metadata<Plato::ScalarMultiVectorT<CurPressT>>(aWorkSets.get("current pressure"));
        auto tPrevPressWS = Plato::metadata<Plato::ScalarMultiVectorT<PrevPressT>>(aWorkSets.get("previous pressure"));
        auto tCriticalTimeStep = Plato::metadata<Plato::ScalarVector>(aWorkSets.get("critical time step"));

        // set local functors
        Plato::ComputeGradientWorkset<mNumSpatialDims> tComputeGradient;
        Plato::InterpolateFromNodal<mNumSpatialDims, mNumDofsPerNode, 0/*offset*/, mNumSpatialDims> tIntrplVectorField;

        // transfer member data to device
        //auto tViscocity = mViscocity;
        auto tPressureTheta = mPressureTheta;
        //auto tViscosityTheta = mViscosityTheta;

        auto tCubWeight = mCubatureRule.getCubWeight();
        auto tBasisFunctions = mCubatureRule.getBasisFunctions();
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
        {
            tComputeGradient(aCellOrdinal, tGradient, tConfigWS, tCellVolume);
            tCellVolume(aCellOrdinal) *= tCubWeight;

            // 1. add previous pressure gradient to residual, i.e. R += Delta{t} G(p_n + \theta\Delta{p})
            Plato::Fluids::calculate_pressure_gradient<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tPressureTheta, tGradient, tCurPressWS, tPrevPressWS, tPressGradGP);
            Plato::Fluids::integrate_vector_field<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tBasisFunctions, tCellVolume, tPressGradGP, aResultWS);
            Plato::blas1::scale<mNumDofsPerCell>(aCellOrdinal, tCriticalTimeStep(0), aResultWS);

            // 2. add viscous force to residual, i.e. R += \theta_3 Ku^{n+1}
            /*Plato::Fluids::strain_rate<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tCurVelWS, tGradient, tCurStrainRate);
            Plato::Fluids::integrate_viscous_forces<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tViscocity, tCellVolume, tGradient, tCurStrainRate, aResultWS, tViscosityTheta);*/

            // 3. add current delta inertial force to residual, i.e. R += M(u_{n+1} - u_n)
            tIntrplVectorField(aCellOrdinal, tBasisFunctions, tCurVelWS, tCurVelGP);
            Plato::Fluids::integrate_vector_field<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tBasisFunctions, tCellVolume, tCurVelGP, aResultWS);
            tIntrplVectorField(aCellOrdinal, tBasisFunctions, tPrevVelWS, tPrevVelGP);
            Plato::Fluids::integrate_vector_field<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tBasisFunctions, tCellVolume, tPrevVelGP, aResultWS, -1.0);

            // 4. add delta predicted inertial force to residual, i.e. R -= M(\bar{u} - u_n)
            tIntrplVectorField(aCellOrdinal, tBasisFunctions, tPredVelWS, tPredVelGP);
            Plato::Fluids::integrate_vector_field<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tBasisFunctions, tCellVolume, tPredVelGP, aResultWS, -1.0);
            Plato::Fluids::integrate_vector_field<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tBasisFunctions, tCellVolume, tPrevVelGP, aResultWS);

            // 5. add stabilizing pressure force to residual, i.e. R += \frac{\Delta{t}^2}{2} Pp^n
            /*auto tMultiplier = static_cast<Plato::Scalar>(0.5)*tCriticalTimeStep(0)*tCriticalTimeStep(0);
            Plato::Fluids::calculate_scalar_field_gradient<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tGradient, tPrevPressWS, tPrevPressGradGP);
            Plato::Fluids::integrate_stabilizing_pressure_gradient<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tCellVolume, tGradient, tPrevVelGP, tPrevPressGradGP, aResultWS, -tMultiplier);*/
        }, "calculate corrected velocity residual");
    }

    void evaluateBoundary
    (const Plato::SpatialModel & aSpatialModel,
     const Plato::WorkSets & aWorkSets,
     Plato::ScalarMultiVectorT<ResultT> & aResult)
    const override
    { return; /* boundary integral equals zero */ }

    void evaluatePrescribed
    (const Plato::SpatialModel & aSpatialModel,
     const Plato::WorkSets & aWorkSets,
     Plato::ScalarMultiVectorT<ResultT> & aResult)
    const override
    { return; /* prescribed force integral equals zero */ }

private:
    void setAritificalPressureDamping(Teuchos::ParameterList& aInputs)
    {
        if(aInputs.isSublist("Time Integration"))
        {
            auto tTimeIntegration = aInputs.sublist("Time Integration");
            mPressureTheta = tTimeIntegration.get<Plato::Scalar>("Pressure Damping", 1.0);
            //mViscosityTheta = tTimeIntegration.get<Plato::Scalar>("Viscosity Damping", 1.0);
        }
    }
};
// class VelocityCorrectorResidual



template<Plato::OrdinalType NumNodes,
         Plato::OrdinalType SpaceDim,
         typename ConfigT,
         typename PrevVelT,
         typename PrevTempT,
         typename ResultT>
DEVICE_TYPE inline void
calculate_convective_forces
(const Plato::OrdinalType & aCellOrdinals,
 const Plato::ScalarArray3DT<ConfigT> & aGradient,
 const Plato::ScalarMultiVectorT<PrevVelT> & aPrevVelGP,
 const Plato::ScalarMultiVectorT<PrevTempT> & aPrevTemp,
 const Plato::ScalarVectorT<ResultT> & aResult,
 Plato::Scalar aMultiplier = 1.0)
{
    for(Plato::OrdinalType tNode = 0; tNode < NumNodes; tNode++)
    {
        for(Plato::OrdinalType tDim = 0; tDim < SpaceDim; tDim++)
        {
            aResult(aCellOrdinals) += aMultiplier * aPrevVelGP(aCellOrdinals, tDim)
                * ( aGradient(aCellOrdinals, tNode, tDim) * aPrevTemp(aCellOrdinals, tNode) );
        }
    }
}

template<Plato::OrdinalType NumNodes,
         typename ConfigT,
         typename SourceT,
         typename ResultT>
DEVICE_TYPE inline void
integrate_scalar_field
(const Plato::OrdinalType & aCellOrdinal,
 const Plato::ScalarVector & aBasisFunctions,
 const Plato::ScalarVectorT<ConfigT> & aCellVolume,
 const Plato::ScalarVectorT<SourceT> & aField,
 const Plato::ScalarMultiVectorT<ResultT> & aResult,
 Plato::Scalar aMultiplier = 1.0)
{
    for(Plato::OrdinalType tNode = 0; tNode < NumNodes; tNode++)
    {
        aResult(aCellOrdinal, tNode) += aMultiplier * aBasisFunctions(tNode) *
            aField(aCellOrdinal) * aCellVolume(aCellOrdinal);
    }
}

template<Plato::OrdinalType NumNodes,
         Plato::OrdinalType SpaceDim,
         typename ConfigT,
         typename SourceT,
         typename ResultT>
DEVICE_TYPE inline void
calculate_flux_divergence
(const Plato::OrdinalType & aCellOrdinal,
 const Plato::ScalarArray3DT<ConfigT> & aGradient,
 const Plato::ScalarVectorT<ConfigT> & aCellVolume,
 const Plato::ScalarMultiVectorT<SourceT> & aFlux,
 const Plato::ScalarMultiVectorT<ResultT> & aResult,
 Plato::Scalar aMultiplier = 1.0)
{
    for(Plato::OrdinalType tNode = 0; tNode < NumNodes; tNode++)
    {
        for(Plato::OrdinalType tDim = 0; tDim < SpaceDim; tDim++)
        {
            aResult(aCellOrdinal, tNode) += aMultiplier * aCellVolume(aCellOrdinal) *
                ( aGradient(aCellOrdinal, tNode, tDim) * aFlux(aCellOrdinal, tDim) );
        }
    }
}

template
<Plato::OrdinalType NumNodes,
 Plato::OrdinalType SpaceDim,
 typename FluxT,
 typename ConfigT,
 typename StateT>
DEVICE_TYPE inline void
calculate_flux
(const Plato::OrdinalType & aCellOrdinal,
 const Plato::ScalarArray3DT<ConfigT> & aGradient,
 const Plato::ScalarMultiVectorT<StateT> & aState,
 const Plato::ScalarMultiVectorT<FluxT> & aFlux)
{
    for(Plato::OrdinalType tNode = 0; tNode < NumNodes; tNode++)
    {
        for(Plato::OrdinalType tDim = 0; tDim < SpaceDim; tDim++)
        {
            aFlux(aCellOrdinal, tDim) += aGradient(aCellOrdinal, tNode, tDim) * aState(aCellOrdinal, tNode);
        }
    }
}


/***************************************************************************//**
 * \tparam NumNodesPerCell number of nodes per cell/element (integer)
 * \tparam ControlT control Forward Automatic Differentiation (FAD) evaluation type
 *
 * \fn DEVICE_TYPE inline ControlT penalize_thermal_diffusivity
 *
 * \brief Penalize thermal diffusivity ratio.
 *
 * \param [in] aCellOrdinal      cell/element ordinal
 * \param [in] aThermalDiffRatio thermal diffusivity ratio (solid diffusivity/fluid diffusivity)
 * \param [in] aPenaltyExponent  SIMP penalty model exponent
 * \param [in] aControl          control work set
 *
 * \return penalized thermal diffusivity ratio
 ******************************************************************************/
template<Plato::OrdinalType NumNodesPerCell,
         typename ControlT>
DEVICE_TYPE inline ControlT
penalize_thermal_diffusivity
(const Plato::OrdinalType & aCellOrdinal,
 const Plato::Scalar & aThermalDiffRatio,
 const Plato::Scalar & aPenaltyExponent,
 const Plato::ScalarMultiVectorT<ControlT> & aControl)
{
    ControlT tDensity = Plato::cell_density<NumNodesPerCell>(aCellOrdinal, aControl);
    ControlT tPenalizedDensity = pow(tDensity, aPenaltyExponent);
    ControlT tPenalizedThermalDiff =
            aThermalDiffRatio + ( (static_cast<Plato::Scalar>(1.0) - aThermalDiffRatio) * tPenalizedDensity);
    return tPenalizedThermalDiff;
}

// TODO: FIX THIS, PUT IT IN TERMS OF APPLY_WEIGHT CLASS
template<Plato::OrdinalType NumNodesPerCell,
         typename ControlT>
DEVICE_TYPE inline ControlT
penalize_heat_source_constant
(const Plato::OrdinalType & aCellOrdinal,
 const Plato::Scalar & aHeatSourceConstant,
 const Plato::Scalar & aPenaltyExponent,
 const Plato::ScalarMultiVectorT<ControlT> & aControl)
{
    ControlT tDensity = Plato::cell_density<NumNodesPerCell>(aCellOrdinal, aControl);
    ControlT tPenalizedDensity = pow(tDensity, aPenaltyExponent);
    auto tPenalizedProperty = (static_cast<Plato::Scalar>(1) - tPenalizedDensity) * aHeatSourceConstant;
    return tPenalizedProperty;
}

template
<Plato::OrdinalType NumNodes,
 Plato::OrdinalType SpaceDim,
 typename ResultT,
 typename ConfigT,
 typename PrevVelT,
 typename StabForceT>
DEVICE_TYPE inline void
integrate_stabilizing_scalar_forces
(const Plato::OrdinalType & aCellOrdinal,
 const Plato::ScalarVectorT<ConfigT> & aCellVolume,
 const Plato::ScalarArray3DT<ConfigT> & aGradient,
 const Plato::ScalarMultiVectorT<PrevVelT> & aPrevVelGP,
 const Plato::ScalarVectorT<StabForceT> & aStabilization,
 const Plato::ScalarMultiVectorT<ResultT> & aResult,
 Plato::Scalar aMultiplier = 1.0)
 {
    for(Plato::OrdinalType tNode = 0; tNode < NumNodes; tNode++)
    {
        for(Plato::OrdinalType tDimK = 0; tDimK < SpaceDim; tDimK++)
        {
            aResult(aCellOrdinal, tNode) += aMultiplier * ( ( aGradient(aCellOrdinal, tNode, tDimK) *
                aPrevVelGP(aCellOrdinal, tDimK) ) * aStabilization(aCellOrdinal) ) * aCellVolume(aCellOrdinal);
        }
    }
 }

template
<Plato::OrdinalType NumNodes,
 typename ResultT,
 typename ConfigT,
 typename CurStateT,
 typename PrevStateT>
DEVICE_TYPE inline void
calculate_inertial_forces
(const Plato::OrdinalType & aCellOrdinal,
 const Plato::ScalarVector & aBasisFunctions,
 const Plato::ScalarVectorT<ConfigT> & aCellVolume,
 const Plato::ScalarVectorT<CurStateT> & aCurrentState,
 const Plato::ScalarVectorT<PrevStateT> & aPreviousState,
 const Plato::ScalarMultiVectorT<ResultT> & aResult)
{
    for (Plato::OrdinalType tNode = 0; tNode < NumNodes; tNode++)
    {
        aResult(aCellOrdinal, tNode) += aCellVolume(aCellOrdinal) *
            aBasisFunctions(tNode) * ( aCurrentState(aCellOrdinal) - aPreviousState(aCellOrdinal) );
    }
}








// todo: calculate average nusset number
template<typename PhysicsT, typename EvaluationT>
class AverageNussetNumber : public Plato::Fluids::AbstractScalarFunction<PhysicsT, EvaluationT>
{
private:
    static constexpr auto mNumSpatialDims = PhysicsT::SimplexT::mNumSpatialDims; /*!< number of spatial dimensions */
    static constexpr auto mNumNodesPerCell = PhysicsT::SimplexT::mNumNodesPerCell; /*!< number of nodes per cell */
    static constexpr auto mNumNodesPerFace = PhysicsT::SimplexT::mNumNodesPerFace; /*!< number of nodes per face */
    static constexpr auto mNumTempDofsPerCell = PhysicsT::SimplexT::mNumEnergyDofsPerCell; /*!< number of energy dofs per cell */
    static constexpr auto mNumTempDofsPerNode = PhysicsT::SimplexT::mNumEnergyDofsPerNode; /*!< number of energy dofs per node */
    static constexpr auto mNumSpatialDimsOnFace = PhysicsT::SimplexT::mNumSpatialDimsOnFace; /*!< number of spatial dimensions on face */

    using ResultT = typename EvaluationT::ResultScalarType; /*!< result FAD type */
    using ConfigT = typename EvaluationT::ConfigScalarType; /*!< configuration FAD type */
    using CurrentTempT = typename EvaluationT::CurrentEnergyScalarType; /*!< current temperature FAD type */
    using ThermalFluxT = typename Plato::Fluids::fad_type_t<typename PhysicsT::SimplexT, CurrentTempT, ConfigT>;

    const Plato::SpatialModel& mSpatialModel; /*!< Plato spatial model */
    Plato::LinearTetCubRuleDegreeOne<mNumSpatialDimsOnFace> mSurfaceCubatureRule; /*!< surface integration rule */

public:
    AverageNussetNumber
    (const Plato::SpatialModel & aModel) :
        mSpatialModel(aModel)
    {}

    std::string name() const override
    {
        return std::string("Average Nusset Number");
    }

    void evaluate
    (const Plato::WorkSets & aWorkSets,
     Plato::ScalarVectorT<ResultT> & aResultWS)
    const override
    {
        // get mesh vertices
        auto tFace2Verts = mSpatialModel.Mesh.ask_verts_of(mNumSpatialDimsOnFace);
        auto tCell2Verts = mSpatialModel.Mesh.ask_elem_verts();

        // get face to element graph
        auto tFace2eElems = mSpatialModel.Mesh.ask_up(mNumSpatialDimsOnFace, mNumSpatialDims);
        auto tFace2Elems_map   = tFace2eElems.a2ab;
        auto tFace2Elems_elems = tFace2eElems.ab2b;

        // get element to face map
        auto tElem2Faces = mSpatialModel.Mesh.ask_down(mNumSpatialDims, mNumSpatialDimsOnFace).ab2b;

        // define local functors
        Plato::ComputeGradientWorkset<mNumSpatialDims> tComputeGradient;
        Plato::CalculateSurfaceArea<mNumSpatialDims> tCalculateSurfaceArea;
        Plato::NodeCoordinate<mNumSpatialDims> tCoords(&(mSpatialModel.Mesh));
        Plato::CalculateSurfaceJacobians<mNumSpatialDims> tCalculateSurfaceJacobians;
        Plato::CreateFaceLocalNode2ElemLocalNodeIndexMap<mNumSpatialDims> tCreateFaceLocalNode2ElemLocalNodeIndexMap;

        // set local containers
        auto tNumCells = mSpatialModel.Mesh.nelems();
        auto tNumFaces = mSpatialModel.Mesh.nfaces();
        Plato::ScalarVectorT<ConfigT> tCellVolume("cell volume", tNumCells);
        Plato::ScalarArray3DT<ConfigT> tGradient("gradient", tNumCells, mNumNodesPerCell, mNumSpatialDims);
        Plato::ScalarArray3DT<ConfigT> tJacobians("jacobian", tNumFaces, mNumSpatialDimsOnFace, mNumSpatialDims);
        Plato::ScalarMultiVectorT<ThermalFluxT> tThermalFlux("current thermal flux", tNumCells, mNumSpatialDims);

        // set input metadata
        auto tConfigWS  = Plato::metadata<Plato::ScalarArray3DT<ConfigT>>(aWorkSets.get("configuration"));
        auto tCurTempWS = Plato::metadata<Plato::ScalarMultiVectorT<CurrentTempT>>(aWorkSets.get("current temperature"));

        auto tCubatureWeight = mSurfaceCubatureRule.getCubWeight();
        auto tBasisFunctions = mSurfaceCubatureRule.getBasisFunctions();
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumFaces), LAMBDA_EXPRESSION(const Plato::OrdinalType & aFaceOrdinal)
        {
            // for each element connected to this face: (either 1 or 2)
            for( Plato::OrdinalType tCell = tFace2Elems_map[aFaceOrdinal]; tCell < tFace2Elems_map[aFaceOrdinal+1]; tCell++ )
            {
                // create map from face local node index to element local node index
                Plato::OrdinalType tLocalNodeOrd[mNumSpatialDims];
                auto tCellOrdinal = tFace2Elems_elems[tCell];
                tCreateFaceLocalNode2ElemLocalNodeIndexMap(tCellOrdinal, aFaceOrdinal, tCell2Verts, tFace2Verts, tLocalNodeOrd);

                // calculate surface area times surface weight
                ConfigT tSurfaceAreaTimesCubWeight(0.0);
                tCalculateSurfaceJacobians(tCellOrdinal, aFaceOrdinal, tLocalNodeOrd, tConfigWS, tJacobians);
                tCalculateSurfaceArea(aFaceOrdinal, tCubatureWeight, tJacobians, tSurfaceAreaTimesCubWeight);

                // calculate thermal flux
                tComputeGradient(tCellOrdinal, tGradient, tConfigWS, tCellVolume);
                Plato::Fluids::calculate_flux<mNumNodesPerCell, mNumSpatialDims>
                    (tCellOrdinal, tGradient, tCurTempWS, tThermalFlux);

                // compute unit normal vector
                auto tElemFaceOrdinal = Plato::get_face_ordinal<mNumSpatialDims>(tCellOrdinal, aFaceOrdinal, tElem2Faces);
                auto tUnitNormalVec = Plato::unit_normal_vector(tCellOrdinal, tElemFaceOrdinal, tCoords);

                // project into aResult workset
                for( Plato::OrdinalType tNode = 0; tNode < mNumNodesPerFace; tNode++)
                {
                    for( Plato::OrdinalType tDim = 0; tDim < mNumSpatialDims; tDim++)
                    {
                        aResultWS(tCellOrdinal) += tBasisFunctions(tNode) * tSurfaceAreaTimesCubWeight * 
			    ( tUnitNormalVec(tDim) * tThermalFlux(tCellOrdinal, tDim) );
                    }
                }
            }
        }, "calculate average nusset number");
    }

    void evaluateBoundary
    (const Plato::WorkSets & aWorkSets,
     Plato::ScalarVectorT<ResultT> & aResult)
    const override
    { return; }
};









// todo: energy equation FINISH DOXYGEN COMMENTS AND CHECK IMPLEMENTATION
template<typename PhysicsT, typename EvaluationT>
class TemperatureResidual : public Plato::Fluids::AbstractVectorFunction<PhysicsT, EvaluationT>
{
private:
    static constexpr auto mNumDofsPerNode = PhysicsT::mNumDofsPerNode;
    static constexpr auto mNumDofsPerCell = PhysicsT::mNumDofsPerCell;

    static constexpr auto mNumSpatialDims       = PhysicsT::SimplexT::mNumSpatialDims;
    static constexpr auto mNumNodesPerCell      = PhysicsT::SimplexT::mNumNodesPerCell;
    static constexpr auto mNumVelDofsPerCell    = PhysicsT::SimplexT::mNumMomentumDofsPerCell;
    static constexpr auto mNumTempDofsPerCell   = PhysicsT::SimplexT::mNumEnergyDofsPerCell;
    static constexpr auto mNumVelDofsPerNode    = PhysicsT::SimplexT::mNumMomentumDofsPerNode;
    static constexpr auto mNumTempDofsPerNode   = PhysicsT::SimplexT::mNumEnergyDofsPerNode;
    static constexpr auto mNumConfigDofsPerCell = PhysicsT::SimplexT::mNumConfigDofsPerCell;

    // set local ad type
    using ResultT   = typename EvaluationT::ResultScalarType;
    using ConfigT   = typename EvaluationT::ConfigScalarType;
    using ControlT  = typename EvaluationT::ControlScalarType;
    using CurVelT   = typename EvaluationT::CurrentMomentumScalarType;
    using CurTempT  = typename EvaluationT::CurrentEnergyScalarType;
    using PrevTempT = typename EvaluationT::PreviousEnergyScalarType;

    using CurFluxT = typename Plato::Fluids::fad_type_t<typename PhysicsT::SimplexT, CurTempT, ConfigT>;
    using PrevFluxT = typename Plato::Fluids::fad_type_t<typename PhysicsT::SimplexT, PrevTempT, ConfigT>;
    using ConvectionT = typename Plato::Fluids::fad_type_t<typename PhysicsT::SimplexT, PrevTempT, CurVelT, ConfigT>;

    Plato::DataMap& mDataMap;
    const Plato::SpatialDomain& mSpatialDomain;

    Plato::LinearTetCubRuleDegreeOne<mNumSpatialDims> mCubatureRule;
    std::shared_ptr<Plato::NaturalBCs<mNumSpatialDims, mNumDofsPerNode>> mHeatFlux;

    Plato::Scalar mTheta = 1.0;
    Plato::Scalar mHeatSourceConstant    = 0.0;
    Plato::Scalar mThermalConductivity   = 1.0;
    Plato::Scalar mCharacteristicLength  = 0.0;
    Plato::Scalar mReferenceTemperature  = 1.0;
    Plato::Scalar mEffectiveConductivity = 1.0;

public:
    TemperatureResidual
    (const Plato::SpatialDomain & aDomain,
     Plato::DataMap             & aDataMap,
     Teuchos::ParameterList     & aInputs) :
         mDataMap(aDataMap),
         mSpatialDomain(aDomain),
         mCubatureRule(Plato::LinearTetCubRuleDegreeOne<mNumSpatialDims>())
    {
        this->setSourceTerm(aInputs);
        this->setDimensionlessProperties(aInputs);
        this->setNaturalBoundaryConditions(aInputs);
        this->setAritificalDiffusiveDamping(aInputs);
    }

    virtual ~TemperatureResidual(){}

    void evaluate
    (const Plato::WorkSets & aWorkSets,
     Plato::ScalarMultiVectorT<ResultT> & aResultWS)
    const override
    {
        auto tNumCells = mSpatialDomain.numCells();
        if( tNumCells != static_cast<Plato::OrdinalType>(aResultWS.extent(0)) )
        {
            THROWERR(std::string("Number of elements mismatch. Spatial domain and output/result workset ")
                + "have different number of cells. " + "Spatial domain has '" + std::to_string(tNumCells)
                + "' elements and output workset has '" + std::to_string(aResultWS.extent(0)) + "' elements.")
        }

        // set constant heat source
        Plato::ScalarVectorT<ResultT> tHeatSource("prescribed heat source", tNumCells);
        Plato::blas1::fill(mHeatSourceConstant, tHeatSource);

        // set local data
        Plato::ScalarVectorT<ConfigT>   tCellVolume("cell weight", tNumCells);
        Plato::ScalarArray3DT<ConfigT>  tGradient("cell gradient", tNumCells, mNumNodesPerCell, mNumSpatialDims);
        Plato::ScalarVectorT<CurTempT>  tCurTempGP("current temperature at Gauss points", tNumCells);
        Plato::ScalarVectorT<PrevTempT> tPrevTempGP("previous temperature at Gauss points", tNumCells);
        Plato::ScalarVectorT<ConvectionT> tConvection("convection", tNumCells);

        Plato::ScalarMultiVectorT<CurFluxT>  tCurThermalFlux("current thermal flux", tNumCells, mNumSpatialDims);
        Plato::ScalarMultiVectorT<PrevFluxT> tPrevThermalFlux("previous thermal flux", tNumCells, mNumSpatialDims);
        Plato::ScalarMultiVectorT<CurVelT>   tCurVelGP("current velocity at Gauss points", tNumCells, mNumVelDofsPerNode);

        // set local functors
        Plato::ComputeGradientWorkset<mNumSpatialDims> tComputeGradient;
        Plato::InterpolateFromNodal<mNumSpatialDims, mNumTempDofsPerNode> tIntrplScalarField;
        Plato::InterpolateFromNodal<mNumSpatialDims, mNumVelDofsPerNode, 0, mNumSpatialDims> tIntrplVectorField;

        // set input state worksets
        auto tConfigWS   = Plato::metadata<Plato::ScalarArray3DT<ConfigT>>(aWorkSets.get("configuration"));
        auto tCurVelWS   = Plato::metadata<Plato::ScalarMultiVectorT<CurVelT>>(aWorkSets.get("current velocity"));
        auto tCurTempWS  = Plato::metadata<Plato::ScalarMultiVectorT<CurTempT>>(aWorkSets.get("current temperature"));
        auto tPrevTempWS = Plato::metadata<Plato::ScalarMultiVectorT<PrevTempT>>(aWorkSets.get("previous temperature"));
        auto tCriticalTimeStep = Plato::metadata<Plato::ScalarVector>(aWorkSets.get("critical time step"));

        // transfer member data to device
        auto tTheta           = mTheta;
        auto tRefTemp         = mReferenceTemperature;
        auto tCharLength      = mCharacteristicLength;
        auto tThermalCond     = mThermalConductivity;
        auto tEffConductivity = mEffectiveConductivity;

        auto tCubWeight = mCubatureRule.getCubWeight();
        auto tBasisFunctions = mCubatureRule.getBasisFunctions();
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
        {
            tComputeGradient(aCellOrdinal, tGradient, tConfigWS, tCellVolume);
            tCellVolume(aCellOrdinal) *= tCubWeight;

            // 1. add previous diffusive force contribution to residual, i.e. R -= (\theta_3-1) K T^n
            auto tMultiplier = (tTheta - static_cast<Plato::Scalar>(1));
            Plato::Fluids::calculate_flux<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tGradient, tPrevTempWS, tPrevThermalFlux);
            Plato::blas1::scale<mNumSpatialDims>(aCellOrdinal, tEffConductivity, tPrevThermalFlux);
            Plato::Fluids::calculate_flux_divergence<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tGradient, tCellVolume, tPrevThermalFlux, aResultWS, -tMultiplier);

            // 2. add previous convective force contribution to residual, i.e. R += C T^n
            tIntrplVectorField(aCellOrdinal, tBasisFunctions, tCurVelWS, tCurVelGP);
            Plato::Fluids::calculate_convective_forces<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tGradient, tCurVelGP, tPrevTempWS, tConvection);
            Plato::Fluids::integrate_scalar_field<mNumTempDofsPerCell>
                (aCellOrdinal, tBasisFunctions, tCellVolume, tConvection, aResultWS);

            // 3. add current diffusive force contribution to residual, i.e. R += \theta_3 K T^{n+1}
            Plato::Fluids::calculate_flux<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tGradient, tCurTempWS, tCurThermalFlux);
            Plato::blas1::scale<mNumSpatialDims>(aCellOrdinal, tEffConductivity, tCurThermalFlux);
            Plato::Fluids::calculate_flux_divergence<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tGradient, tCellVolume, tCurThermalFlux, aResultWS, tTheta);

            // 4. add previous heat source contribution to residual, i.e. R -= \alpha Q^n
            auto tHeatSourceCnst = ( tCharLength * tCharLength ) / (tThermalCond * tRefTemp);
            Plato::Fluids::integrate_scalar_field<mNumTempDofsPerCell>
                (aCellOrdinal, tBasisFunctions, tCellVolume, tHeatSource, aResultWS, -tHeatSourceCnst);

            // 5. apply time step, i.e. R = \Delta{t}*( \theta_3 K T^{n+1} + C T^n - (\theta_3-1) K T^n - Q^n)
            Plato::blas1::scale<mNumTempDofsPerCell>(aCellOrdinal, tCriticalTimeStep(0), aResultWS);

            // 6. add previous inertial force contribution to residual, i.e. R -= M T^n
            tIntrplVectorField(aCellOrdinal, tBasisFunctions, tPrevTempWS, tPrevTempGP);
            Plato::Fluids::integrate_scalar_field<mNumNodesPerCell>
                (aCellOrdinal, tBasisFunctions, tCellVolume, tPrevTempGP, aResultWS, -1.0);

            // 7. add current inertial force contribution to residual, i.e. R += M T^{n+1}
            tIntrplVectorField(aCellOrdinal, tBasisFunctions, tCurTempWS, tCurTempGP);
            Plato::Fluids::integrate_scalar_field<mNumNodesPerCell>
                (aCellOrdinal, tBasisFunctions, tCellVolume, tCurTempGP, aResultWS);
        }, "energy conservation residual");
    }

    void evaluateBoundary
    (const Plato::SpatialModel & aSpatialModel,
     const Plato::WorkSets & aWorkSets,
     Plato::ScalarMultiVectorT<ResultT> & aResult)
    const override
    { return;  }

    void evaluatePrescribed
    (const Plato::SpatialModel & aSpatialModel,
     const Plato::WorkSets & aWorkSets,
     Plato::ScalarMultiVectorT<ResultT> & aResultWS)
    const override
    {
        if( mHeatFlux != nullptr )
        {
            // set input state worksets
            auto tConfigWS   = Plato::metadata<Plato::ScalarArray3DT<ConfigT>>(aWorkSets.get("configuration"));
            auto tControlWS  = Plato::metadata<Plato::ScalarMultiVectorT<ControlT>>(aWorkSets.get("control"));
            auto tPrevTempWS = Plato::metadata<Plato::ScalarMultiVectorT<PrevTempT>>(aWorkSets.get("previous temperature"));

            // evaluate prescribed flux
            auto tNumCells = aResultWS.extent(0);
            Plato::ScalarMultiVectorT<ResultT> tHeatFluxWS("heat flux", tNumCells, mNumDofsPerCell);
            mHeatFlux->get( aSpatialModel, tPrevTempWS, tControlWS, tConfigWS, tHeatFluxWS );

            auto tCriticalTimeStep = Plato::metadata<Plato::ScalarVector>(aWorkSets.get("critical time step"));
            Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
            {
                Plato::blas1::scale<mNumDofsPerCell>(aCellOrdinal, tCriticalTimeStep(0), tHeatFluxWS);
                Plato::blas1::update<mNumDofsPerCell>(aCellOrdinal, -1.0, tHeatFluxWS, 1.0, aResultWS);
            }, "heat flux contribution");
        }
    }

private:
    void setSourceTerm
    (Teuchos::ParameterList & aInputs)
    {
        if(aInputs.isSublist("Heat Source"))
        {
            auto tHeatSource = aInputs.sublist("Heat Source");
            mHeatSourceConstant = tHeatSource.get<Plato::Scalar>("Constant", 0.0);
            mReferenceTemperature = tHeatSource.get<Plato::Scalar>("Reference Temperature", 1.0);
            if(mReferenceTemperature == static_cast<Plato::Scalar>(0.0))
            {
                THROWERR(std::string("Invalid 'Reference Temperature' input, value is set to an invalid numeric number '")
                    + std::to_string(mReferenceTemperature) + "'.")
            }

            this->setThermalProperties(aInputs);
            this->setCharacteristicLength(aInputs);
        }
    }

    void setThermalProperties
    (Teuchos::ParameterList & aInputs)
    {
        if(aInputs.isSublist("Heat Source"))
        {
            auto tMaterialName = mSpatialDomain.getMaterialName();
            Plato::is_material_defined(tMaterialName, aInputs);
            auto tMaterial = aInputs.sublist("Material Models").sublist(tMaterialName);
            auto tThermalPropBlock = std::string("Thermal Properties");
            mThermalConductivity = Plato::parse_parameter<Plato::Scalar>("Thermal Conductivity", tThermalPropBlock, tMaterial);
            if(mThermalConductivity <= static_cast<Plato::Scalar>(0.0))
            {
                THROWERR(std::string("Invalid 'Thermal Conductivity' input, value is set to an invalid numeric number '")
                    + std::to_string(mThermalConductivity) + "'.")
            }
        }
    }

    void setCharacteristicLength
    (Teuchos::ParameterList & aInputs)
    {
        if(aInputs.isSublist("Hyperbolic") == false)
        {
            THROWERR("'Hyperbolic' Parameter List is not defined.")
        }
        auto tHyperbolic = aInputs.sublist("Hyperbolic");
        mCharacteristicLength = Plato::parse_parameter<Plato::Scalar>("Characteristic Length", "Dimensionless Properties", tHyperbolic);
    }

    void setEffectiveConductivity
    (Teuchos::ParameterList & aInputs)
    {
        auto tHyperbolic = aInputs.sublist("Hyperbolic");
        auto tTag = tHyperbolic.get<std::string>("Heat Transfer", "natural");
        auto tHeatTransfer = Plato::tolower(tTag);

        if(tHeatTransfer == "forced" || tHeatTransfer == "mixed")
        {
            auto tPrNum = Plato::parse_parameter<Plato::Scalar>("Prandtl Number", "Dimensionless Properties", tHyperbolic);
            auto tReNum = Plato::parse_parameter<Plato::Scalar>("Reynolds Number", "Dimensionless Properties", tHyperbolic);
            mEffectiveConductivity = static_cast<Plato::Scalar>(1) / (tReNum*tPrNum);
        }
        else if(tHeatTransfer == "natural")
        {
            mEffectiveConductivity = 1.0;
        }
        else
        {
            THROWERR(std::string("'Heat Transfer' mechanism with tag '") + tHeatTransfer + "' is not supported.")
        }
    }

    void setDimensionlessProperties
    (Teuchos::ParameterList & aInputs)
    {
        if(aInputs.isSublist("Hyperbolic") == false)
        {
            THROWERR("'Hyperbolic' Parameter List is not defined.")
        }
        this->setEffectiveConductivity(aInputs);
    }

    void setAritificalDiffusiveDamping
    (Teuchos::ParameterList& aInputs)
    {
        if(aInputs.isSublist("Time Integration"))
        {
            auto tTimeIntegration = aInputs.sublist("Time Integration");
            mTheta = tTimeIntegration.get<Plato::Scalar>("Diffusive Damping", 1.0);
        }
    }

    void setNaturalBoundaryConditions
    (Teuchos::ParameterList & aInputs)
    {
        if(aInputs.isSublist("Energy Natural Boundary Conditions"))
        {
            auto tSublist = aInputs.sublist("Energy Natural Boundary Conditions");
            mHeatFlux = std::make_shared<Plato::NaturalBCs<mNumSpatialDims, mNumDofsPerNode>>(tSublist);
        }
    }
};


// todo: energy equation - to
namespace SIMP
{

/***************************************************************************//**
 * \class TemperatureResidual
 *
 * \tparam PhysicsT    physics type
 * \tparam EvaluationT forward automatic differentiation evaluation type
 *
 * \brief Evaluate energy conservation (i.e. temperature equation) residual, which
 * is given by
 *
 * \f[
 *   \mathcal{R}^n(q^h) = I^n(q^h) - F^n(q^h) - S^n(q^h) - E^n(q^h) = 0.
 * \f]
 *
 * where
 *
 * Inertial:
 *
 * \f[
 *   I^n(q^h) = \int_{\Omega}q^h \left(\frac{\bar{T}^{n+1} - \bar{T}^{\ast}}{\Delta\bar{t}}\right)d\Omega
 * \f]
 *
 * Internal:
 *
 * \f[
 *   F^n(q^h) =
 *     - \int_{\Omega}q^h\left(\bar{u}^n_i\frac{\partial\bar{T}^n}{\partial\bar{x}_i}\right)d\Omega
 *     - \int_{\Omega}\frac{\partial q^h}{\partial\bar{x}_i}\left(\pi^{\alpha}(\theta)
 *       \frac{\partial\bar{T}}{\partial\bar{x}_i}\right)d\Omega
 *     + \int_{\Omega}q^h\left(\pi^{\beta}(\theta)Q\right)d\Omega
 * \f]
 *
 * Here, the material penalty function \f$ \pi^{\alpha}(\theta) \f$ is introduced to
 * allow penalization of the thermal diffusivity \f$ \alpha \f$ for density-based
 * topology optimization. The variable \f$ \theta \f$ denotes the density field used
 * to define the geometry parameterization. Similarly, the volumetric heat source
 * is penalized by \f$ \pi^{\beta}(\theta) \f$ within the fluid domain since only
 * the solid domain is assumed to conduct heat.
 *
 * External:
 *
 * \f[
 *   E_i^n(w^h_i) = \int_{\Gamma_H}q^h H\,d\Gamma
 * \f]
 *
 * Stabilizing:
 *
 * \f[
 *   S^n(q^h) =
 *     \frac{\Delta\bar{t}}{2}\int_{\Omega}\left( \frac{\partial q^h}{\partial\bar{x}_k}\bar{u}^n_k
 *     + q^h\frac{\partial\bar{u}^n_k}{\partial\bar{x}_k} \right)\hat{R}^n_T\, d\Omega
 * \f]
 *
 * where
 *
 * \f[
 *   \hat{R}^n_T = -\bar{u}_i\frac{\partial\bar{T}}{\partial\bar{x}_i} + \pi^{\beta}(\theta)Q
 * \f]
 *
 * The variable \f$ Q \f$ denotes a volumetric heat source.
 *
 ******************************************************************************/
template<typename PhysicsT, typename EvaluationT>
class TemperatureResidual : public Plato::Fluids::AbstractVectorFunction<PhysicsT, EvaluationT>
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
    using DivergenceT   = typename Plato::Fluids::fad_type_t<typename PhysicsT::SimplexT, PrevVelT, ConfigT>;

    Plato::DataMap& mDataMap;                   /*!< output database */
    const Plato::SpatialDomain& mSpatialDomain; /*!< Plato spatial model */

    Plato::LinearTetCubRuleDegreeOne<mNumSpatialDims> mCubatureRule; /*!< integration rule */
    std::shared_ptr<Plato::NaturalBCs<mNumSpatialDims, mNumDofsPerNode>> mHeatFlux; /*!< heat flux evaluator */

    Plato::Scalar mHeatSourceConstant         = 0.0;
    Plato::Scalar mThermalConductivity        = 1.0;
    Plato::Scalar mCharacteristicLength       = 1.0;
    Plato::Scalar mReferenceTemperature       = 1.0;
    Plato::Scalar mEffectiveConductivity      = 1.0;
    Plato::Scalar mThermalDiffusivityRatio    = 1.0; /*!< thermal diffusivity ratio, e.g. solid diffusivity / fluid diffusivity */
    Plato::Scalar mHeatSourcePenaltyExponent  = 3.0;
    Plato::Scalar mThermalDiffPenaltyExponent = 3.0;

public:
    TemperatureResidual
    (const Plato::SpatialDomain & aDomain,
     Plato::DataMap             & aDataMap,
     Teuchos::ParameterList     & aInputs) :
         mDataMap(aDataMap),
         mSpatialDomain(aDomain),
         mCubatureRule(Plato::LinearTetCubRuleDegreeOne<mNumSpatialDims>())
    {
        this->setSourceTerm(aInputs);
        this->setThermalProperties(aInputs);
        this->setPenaltyModelParameters(aInputs);
        this->setDimensionlessProperties(aInputs);
        this->setNaturalBoundaryConditions(aInputs);
    }

    virtual ~TemperatureResidual(){}

    /***************************************************************************//**
     * \fn void evaluate
     *
     * \brief Evaluate internal pressure residual, which is defined by
     *
     * \f[
     *   \hat{R}^n(v^h) = I^n(v^h) - F^n(v^h) = 0.
     * \f]
     *
     * Inertial Forces:
     *
     * \f[
     *   I^n(q^h) = \int_{\Omega}q^h \left(\frac{\bar{T}^{n+1} - \bar{T}^{\ast}}{\Delta\bar{t}}\right)d\Omega
     * \f]
     *
     * Internal Forces:
     *
     * \f[
     *   F^n(q^h) =
     *     - \int_{\Omega}q^h\left(\bar{u}^n_i\frac{\partial\bar{T}^n}{\partial\bar{x}_i}\right)d\Omega
     *     - \int_{\Omega}\frac{\partial q^h}{\partial\bar{x}_i}\left(\pi^{\alpha}(\theta)
     *       \frac{\partial\bar{T}}{\partial\bar{x}_i}\right)d\Omega
     *     + \int_{\Omega}q^h\left(\pi^{\beta}(\theta)Q\right)d\Omega
     * \f]
     *
     ******************************************************************************/
    void evaluate
    (const Plato::WorkSets & aWorkSets,
     Plato::ScalarMultiVectorT<ResultT> & aResultWS)
    const override
    {
        auto tNumCells = mSpatialDomain.numCells();
        if( tNumCells != static_cast<Plato::OrdinalType>(aResultWS.extent(0)) )
        {
            THROWERR(std::string("Number of elements mismatch. Spatial domain and output/result workset ")
                + "have different number of cells. " + "Spatial domain has '" + std::to_string(tNumCells)
                + "' elements and output workset has '" + std::to_string(aResultWS.extent(0)) + "' elements.")
        }

        // set constant heat source
        Plato::ScalarVectorT<ResultT> tPrescribedHeatSource("prescribed heat source", tNumCells);
        Plato::blas1::fill(mHeatSourceConstant, tPrescribedHeatSource);

        // set local data
        Plato::ScalarVectorT<ConfigT>   tCellVolume("cell weight", tNumCells);
        Plato::ScalarVectorT<CurTempT>  tCurTempGP("current temperature at Gauss points", tNumCells);
        Plato::ScalarVectorT<PrevTempT> tPrevTempGP("previous temperature at Gauss points", tNumCells);

        Plato::ScalarVectorT<ResultT> tConvection("conduction", tNumCells);
        Plato::ScalarVectorT<ResultT> tHeatSource("heat source", tNumCells);
        Plato::ScalarVectorT<ResultT> tStabilization("stabilization", tNumCells);

        Plato::ScalarMultiVectorT<ResultT>  tThermalFlux("thermal flux", tNumCells, mNumSpatialDims);
        Plato::ScalarMultiVectorT<ResultT>  tStabForces("stabilizing forces", tNumCells, mNumTempDofsPerCell);
        Plato::ScalarMultiVectorT<ResultT>  tInternalForces("internal forces", tNumCells, mNumTempDofsPerCell);
        Plato::ScalarMultiVectorT<PrevVelT> tPrevVelGP("previous velocity at Gauss points", tNumCells, mNumVelDofsPerNode);

        Plato::ScalarArray3DT<ConfigT> tGradient("cell gradient", tNumCells, mNumNodesPerCell, mNumSpatialDims);

        // set local functors
        Plato::ComputeGradientWorkset<mNumSpatialDims> tComputeGradient;
        Plato::InterpolateFromNodal<mNumSpatialDims, mNumTempDofsPerNode> tIntrplScalarField;
        Plato::InterpolateFromNodal<mNumSpatialDims, mNumVelDofsPerNode, 0/*offset*/, mNumSpatialDims> tIntrplVectorField;

        // set input state worksets
        auto tControlWS  = Plato::metadata<Plato::ScalarMultiVectorT<ControlT>>(aWorkSets.get("control"));
        auto tConfigWS   = Plato::metadata<Plato::ScalarArray3DT<ConfigT>>(aWorkSets.get("configuration"));
        auto tPrevVelWS  = Plato::metadata<Plato::ScalarMultiVectorT<PrevVelT>>(aWorkSets.get("previous velocity"));
        auto tCurTempWS  = Plato::metadata<Plato::ScalarMultiVectorT<CurTempT>>(aWorkSets.get("current temperature"));
        auto tPrevTempWS = Plato::metadata<Plato::ScalarMultiVectorT<PrevTempT>>(aWorkSets.get("previous temperature"));
        auto tCriticalTimeStep = Plato::metadata<Plato::ScalarVector>(aWorkSets.get("critical time step"));

        // transfer member data to device
        auto tRefTemp               = mReferenceTemperature;
        auto tCharLength            = mCharacteristicLength;
        auto tThermalCond           = mThermalConductivity;
        auto tEffConductivity       = mEffectiveConductivity;
        auto tThermalDiffRatio      = mThermalDiffusivityRatio;
        auto tHeatSrcPenaltyExp     = mHeatSourcePenaltyExponent;
        auto tThermalDiffPenaltyExp = mThermalDiffPenaltyExponent;

        auto tCubWeight = mCubatureRule.getCubWeight();
        auto tBasisFunctions = mCubatureRule.getBasisFunctions();
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
        {
            tComputeGradient(aCellOrdinal, tGradient, tConfigWS, tCellVolume);
            tCellVolume(aCellOrdinal) *= tCubWeight;

            // 1. calculate internal forces
            ControlT tPenalizedDiffusivityRatio = Plato::Fluids::penalize_thermal_diffusivity<mNumNodesPerCell>
                (aCellOrdinal, tThermalDiffRatio, tThermalDiffPenaltyExp, tControlWS);
            tPenalizedDiffusivityRatio = tEffConductivity * tPenalizedDiffusivityRatio;
            Plato::Fluids::calculate_flux<mNumNodesPerCell,mNumSpatialDims>
                (aCellOrdinal, tGradient, tPrevTempWS, tThermalFlux);
            Plato::blas1::scale<mNumSpatialDims>(aCellOrdinal, tPenalizedDiffusivityRatio, tThermalFlux);
            Plato::Fluids::calculate_flux_divergence<mNumNodesPerCell,mNumSpatialDims>
                (aCellOrdinal, tGradient, tCellVolume, tThermalFlux, tInternalForces, -1.0);

            tIntrplVectorField(aCellOrdinal, tBasisFunctions, tPrevVelWS, tPrevVelGP);
            Plato::Fluids::calculate_convective_forces<mNumNodesPerCell,mNumSpatialDims>
                (aCellOrdinal, tGradient, tPrevVelGP, tPrevTempWS, tConvection);
            Plato::Fluids::integrate_scalar_field<mNumTempDofsPerCell>
                (aCellOrdinal, tBasisFunctions, tCellVolume, tConvection, tInternalForces, -1.0);

            auto tDimensionlessConst = ( tCharLength * tCharLength ) / (tThermalCond * tRefTemp);
            ControlT tPenalizedDimensionlessConst = Plato::Fluids::penalize_heat_source_constant<mNumNodesPerCell>
                (aCellOrdinal, tDimensionlessConst, tHeatSrcPenaltyExp, tControlWS);
            tHeatSource(aCellOrdinal) += tPenalizedDimensionlessConst * tPrescribedHeatSource(aCellOrdinal);
            Plato::Fluids::integrate_scalar_field<mNumNodesPerCell>
                (aCellOrdinal, tBasisFunctions, tCellVolume, tHeatSource, tInternalForces);
            Plato::blas1::scale<mNumDofsPerCell>(aCellOrdinal, tCriticalTimeStep(0), tInternalForces);

            // 2. calculate stabilizing forces
            tStabilization(aCellOrdinal) += tHeatSource(aCellOrdinal) - tConvection(aCellOrdinal);
            Plato::Fluids::integrate_stabilizing_scalar_forces<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tCellVolume, tGradient, tPrevVelGP, tStabilization, tStabForces);
            auto tMultiplier = static_cast<Plato::Scalar>(0.5) * tCriticalTimeStep(0) * tCriticalTimeStep(0);
            Plato::blas1::scale<mNumDofsPerCell>(aCellOrdinal, tMultiplier, tInternalForces);

            // 3. add inertial force contribution
            tIntrplVectorField(aCellOrdinal, tBasisFunctions, tCurTempWS, tCurTempGP);
            tIntrplVectorField(aCellOrdinal, tBasisFunctions, tPrevTempWS, tPrevTempGP);
            Plato::Fluids::calculate_inertial_forces<mNumNodesPerCell>
                (aCellOrdinal, tBasisFunctions, tCellVolume, tCurTempGP, tPrevTempGP, aResultWS);
            Plato::blas1::update<mNumDofsPerCell>(aCellOrdinal, -1.0, tStabForces, 1.0, aResultWS);
            Plato::blas1::update<mNumDofsPerCell>(aCellOrdinal, -1.0, tInternalForces, 1.0, aResultWS);
        }, "temperature residual");
    }

    void evaluateBoundary
    (const Plato::SpatialModel & aSpatialModel,
     const Plato::WorkSets & aWorkSets,
     Plato::ScalarMultiVectorT<ResultT> & aResult)
    const override
    { return; /* boundary integral equates zero */ }

    /***************************************************************************//**
     * \fn void evaluatePrescribed
     *
     * \brief Evaluate prescribed heat flux, which is defined by
     *
     * \f[
     *   E_i^n(w^h_i) = \int_{\Gamma_H}q^h H\,d\Gamma
     * \f]
     *
     * Recall, the residual equation is being evaluated, i.e.
     *
     * \f[
     *   \mathcal{R}^n(q^h) = I^n(q^h) - F^n(q^h) - S^n(q^h) - E^n(q^h) = 0.
     * \f]
     *
     * Therefore, the prescribed heat flux is multiplied by -1.0.
     *
     ******************************************************************************/
    void evaluatePrescribed
    (const Plato::SpatialModel & aSpatialModel,
     const Plato::WorkSets & aWorkSets,
     Plato::ScalarMultiVectorT<ResultT> & aResultWS)
    const override
    {
        if( mHeatFlux != nullptr )
        {
            // set input state worksets
            auto tConfigWS   = Plato::metadata<Plato::ScalarArray3DT<ConfigT>>(aWorkSets.get("configuration"));
            auto tControlWS  = Plato::metadata<Plato::ScalarMultiVectorT<ControlT>>(aWorkSets.get("control"));
            auto tPrevTempWS = Plato::metadata<Plato::ScalarMultiVectorT<PrevTempT>>(aWorkSets.get("previous temperature"));

            // evaluate prescribed flux
            auto tNumCells = aResultWS.extent(0);
            Plato::ScalarMultiVectorT<ResultT> tResultWS("heat flux", tNumCells, mNumDofsPerCell);
            mHeatFlux->get( aSpatialModel, tPrevTempWS, tControlWS, tConfigWS, tResultWS );

            auto tCriticalTimeStep = Plato::metadata<Plato::ScalarVector>(aWorkSets.get("critical time step"));
            Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
            {
                Plato::blas1::scale<mNumDofsPerCell>(aCellOrdinal, tCriticalTimeStep(0), aResultWS);
                Plato::blas1::update<mNumDofsPerCell>(aCellOrdinal, -1.0, tResultWS, 1.0, aResultWS);
            }, "heat flux contribution");
        }
    }

private:
    void setSourceTerm
    (Teuchos::ParameterList & aInputs)
    {
        if(aInputs.isSublist("Heat Source"))
        {
            auto tHeatSource = aInputs.sublist("Heat Source");
            mHeatSourceConstant = tHeatSource.get<Plato::Scalar>("Constant", 0.0);
            mReferenceTemperature = tHeatSource.get<Plato::Scalar>("Reference Temperature", 0.0);
            this->setCharacteristicLength(aInputs);
        }
    }

    void setCharacteristicLength
    (Teuchos::ParameterList & aInputs)
    {
        if(!aInputs.isSublist("Hyperbolic"))
        {
            THROWERR("'Hyperbolic' Parameter List is not defined.")
        }
        auto tHyperbolic = aInputs.sublist("Hyperbolic");
        mCharacteristicLength = Plato::parse_parameter<Plato::Scalar>("Characteristic Length", "Dimensionless Properties", tHyperbolic);
    }

    void setThermalProperties
    (Teuchos::ParameterList & aInputs)
    {
        auto tMaterialName = mSpatialDomain.getMaterialName();
        Plato::is_material_defined(tMaterialName, aInputs);
        auto tMaterial = aInputs.sublist("Material Models").sublist(tMaterialName);
        auto tThermalPropBlock = std::string("Thermal Properties");
        mThermalConductivity = Plato::parse_parameter<Plato::Scalar>("Thermal Conductivity", tThermalPropBlock, tMaterial);
        mThermalDiffusivityRatio = Plato::parse_parameter<Plato::Scalar>("Thermal Diffusivity Ratio", tThermalPropBlock, tMaterial);
    }

    void setEffectiveConductivity
    (Teuchos::ParameterList & aInputs)
    {
        auto tHyperbolic = aInputs.sublist("Hyperbolic");
        auto tTag = tHyperbolic.get<std::string>("Heat Transfer", "None");
        auto tHeatTransfer = Plato::tolower(tTag);

        if(tHeatTransfer == "forced" || tHeatTransfer == "none")
        {
            auto tPrNum = Plato::parse_parameter<Plato::Scalar>("Prandtl Number", "Dimensionless Properties", tHyperbolic);
            auto tReNum = Plato::parse_parameter<Plato::Scalar>("Reynolds Number", "Dimensionless Properties", tHyperbolic);
            mEffectiveConductivity = static_cast<Plato::Scalar>(1) / (tReNum*tPrNum);
        }
        else if(tHeatTransfer == "natural")
        {
            mEffectiveConductivity = 1.0;
        }
        else
        {
            THROWERR(std::string("'Heat Transfer' mechanism with tag '") + tHeatTransfer + "' is not supported.")
        }
    }

    void setDimensionlessProperties
    (Teuchos::ParameterList & aInputs)
    {
        if(aInputs.isSublist("Hyperbolic") == false)
        {
            THROWERR("'Hyperbolic' Parameter List is not defined.")
        }
        this->setEffectiveConductivity(aInputs);
    }

    void setPenaltyModelParameters
    (Teuchos::ParameterList & aInputs)
    {
        if(aInputs.isSublist("Hyperbolic") == false)
        {
            THROWERR("'Hyperbolic' Parameter List is not defined.")
        }
        auto tHyperbolicParamList = aInputs.sublist("Hyperbolic");
        if(tHyperbolicParamList.isSublist("Energy Conservation"))
        {
            auto tEnergyParamList = tHyperbolicParamList.sublist("Energy Conservation");
            if (tEnergyParamList.isSublist("Penalty Function"))
            {
                auto tPenaltyFuncList = tEnergyParamList.sublist("Penalty Function");
                mHeatSourcePenaltyExponent = tPenaltyFuncList.get<Plato::Scalar>("Heat Source Penalty Exponent", 3.0);
                mThermalDiffPenaltyExponent = tPenaltyFuncList.get<Plato::Scalar>("Thermal Diffusion Penalty Exponent", 3.0);
            }
        }
    }

    void setNaturalBoundaryConditions
    (Teuchos::ParameterList & aInputs)
    {
        if(aInputs.isSublist("Energy Natural Boundary Conditions"))
        {
            auto tSublist = aInputs.sublist("Energy Natural Boundary Conditions");
            mHeatFlux = std::make_shared<Plato::NaturalBCs<mNumSpatialDims, mNumDofsPerNode>>(tSublist);
        }
    }
};
// class TemperatureResidual

}
// namespace SIMP




/***************************************************************************//**
 * \fn device_type void integrate_divergence_operator
 *
 * \brief Integrate momentum divergence, which is defined as
 *
 * \f[
 *   \alpha\int_{\Omega} v^h\frac{\partial\bar{u}_i^{n}}{\partial\bar{x}_i} d\Omega
 * \f]
 *
 * where \f$ v^h \f$ is the test function, \f$ \bar{u}_i^{n} \f$ is the previous
 * velocity, and \f$ \alpha \f$ denotes a scalar multiplier.
 *
 ******************************************************************************/
template<Plato::OrdinalType NumNodesPerCell,
         Plato::OrdinalType NumSpatialDims,
         typename ConfigT,
         typename PrevVelT,
         typename ResultT>
DEVICE_TYPE inline void
integrate_divergence_operator
(const Plato::OrdinalType & aCellOrdinal,
 const Plato::ScalarVector & aBasisFunctions,
 const Plato::ScalarArray3DT<ConfigT> & aGradient,
 const Plato::ScalarVectorT<ConfigT> & aCellVolume,
 const Plato::ScalarMultiVectorT<PrevVelT> & aPrevVel,
 const Plato::ScalarMultiVectorT<ResultT> & aResult,
 Plato::Scalar aMultiplier = 1.0)
{
    for(Plato::OrdinalType tNode = 0; tNode < NumNodesPerCell; tNode++)
    {
        for(Plato::OrdinalType tDim = 0; tDim < NumSpatialDims; tDim++)
        {
            aResult(aCellOrdinal, tNode) += aMultiplier * aCellVolume(aCellOrdinal) *
                aBasisFunctions(tNode) * aGradient(aCellOrdinal, tNode, tDim) * aPrevVel(aCellOrdinal, tDim);
        }
    }
}

/***************************************************************************//**
 * \fn device_type void integrate_divergence_delta_predicted_momentum
 *
 * \brief Integrate divergence predicted momentum increment, which is defined as
 *
 * \f[
 *   \theta_1\int_{\Omega} \frac{\partial v^h}{\partial\bar{x}_i}\Delta{u}^{\ast}_i d\Omega
 * \f]
 *
 * where \f$ \Delta{u}^{\ast}_i = \bar{u}^{\ast}_i - \bar{u}^{n-1}_i. \f$
 *
 * Here, \f$ v^h \f$ is the test function, \f$ \bar{u}_i^{n} \f$ is the previous
 * velocity, \f$ \bar{u}_i^{\ast} \f$ is the current predicted velocity, and
 * \f$ \theta_1 \f$ denotes a scalar multiplier.
 *
 ******************************************************************************/
template<Plato::OrdinalType NumNodesPerCell,
         Plato::OrdinalType NumSpatialDims,
         typename ConfigT,
         typename PrevVelT,
         typename PredVelT,
         typename ResultT>
DEVICE_TYPE inline void
integrate_divergence_delta_predicted_momentum
(const Plato::OrdinalType & aCellOrdinal,
 const Plato::ScalarArray3DT<ConfigT> & aGradient,
 const Plato::ScalarVectorT<ConfigT> & aCellVolume,
 const Plato::ScalarMultiVectorT<PredVelT> & aPredVelGP,
 const Plato::ScalarMultiVectorT<PrevVelT> & aPrevVelGP,
 const Plato::ScalarMultiVectorT<ResultT> & aResult,
 Plato::Scalar aMultiplier = 1.0)
{
    for(Plato::OrdinalType tNode = 0; tNode < NumNodesPerCell; tNode++)
    {
        for(Plato::OrdinalType tDim = 0; tDim < NumSpatialDims; tDim++)
        {
            aResult(aCellOrdinal, tNode) += aMultiplier * aCellVolume(aCellOrdinal) *
                aGradient(aCellOrdinal, tNode, tDim) * ( aPredVelGP(aCellOrdinal, tDim) -
                    aPrevVelGP(aCellOrdinal, tDim) );
        }
    }
}

template<Plato::OrdinalType NumNodesPerCell,
         Plato::OrdinalType NumSpatialDims,
         typename ConfigT,
         typename PressT,
         typename ResultT>
DEVICE_TYPE inline void
integrate_laplacian_operator
(const Plato::OrdinalType & aCellOrdinal,
 const Plato::ScalarArray3DT<ConfigT> & aGradient,
 const Plato::ScalarVectorT<ConfigT> & aCellVolume,
 const Plato::ScalarMultiVectorT<PressT> & aVecField,
 const Plato::ScalarMultiVectorT<ResultT> & aResult,
 Plato::Scalar aMultiplier = 1.0)
{
    for(Plato::OrdinalType tNode = 0; tNode < NumNodesPerCell; tNode++)
    {
        for(Plato::OrdinalType tDim = 0; tDim < NumSpatialDims; tDim++)
        {
            aResult(aCellOrdinal, tNode) += aMultiplier * aCellVolume(aCellOrdinal) *
                aGradient(aCellOrdinal, tNode, tDim) * aVecField(aCellOrdinal, tDim);
        }
    }
}

/***************************************************************************//**
 * \fn device_type void integrate_divergence_previous_pressure_gradient
 *
 * \brief Integrate divergence previous pressure gradient, which is defined as
 *
 * \f[
 *   \theta_1\int_{\Omega_e} \Delta{t}\frac{\partial v^h}{partial x_i}
 *     \frac{\partial p^{n}}{\partial x_i} d\Omega,
 * \f]
 *
 * where \f$ v^h \f$ is the test function, \f$ p^{n} \f$ is the previous
 * pressure, \f$ \Delta{t} \f$ is the current nodal time step, and
 * \f$ \theta_1 \f$ denotes a scalar multiplier.
 *
 ******************************************************************************/
template<Plato::OrdinalType NumNodesPerCell,
         Plato::OrdinalType NumSpatialDims,
         typename ConfigT,
         typename PrevPressT,
         typename ResultT>
DEVICE_TYPE inline void
integrate_divergence_previous_pressure_gradient
(const Plato::OrdinalType & aCellOrdinal,
 const Plato::ScalarMultiVector & aTimeStep,
 const Plato::ScalarArray3DT<ConfigT> & aGradient,
 const Plato::ScalarVectorT<ConfigT> & aCellVolume,
 const Plato::ScalarMultiVectorT<PrevPressT> & aPrevPressGrad,
 const Plato::ScalarMultiVectorT<ResultT> & aResult,
 Plato::Scalar aMultiplier = 1.0)
{
    for(Plato::OrdinalType tNode = 0; tNode < NumNodesPerCell; tNode++)
    {
        for(Plato::OrdinalType tDim = 0; tDim < NumSpatialDims; tDim++)
        {
            aResult(aCellOrdinal, tNode) += aMultiplier * aTimeStep(aCellOrdinal, tNode) *
                aGradient(aCellOrdinal, tNode, tDim) * aPrevPressGrad(aCellOrdinal, tDim) *
                aCellVolume(aCellOrdinal);
        }
    }
}

/***************************************************************************//**
 * \fn device_type void integrate_divergence_delta_pressure_gradient
 *
 * \brief Integrate divergence delta pressure gradient, which is defined as
 *
 * \f[
 *   \alpha\int_{\Omega_e} \frac{\partial v^h}{partial x_i}\Delta{t}
 *     \frac{\partial\Delta{p}}{\partial x_i} d\Omega,
 * \f]
 *
 * where \f$ v^h \f$ is the test function, \f$ \Delta{p}=p^{n+1}-p^{n} \f$,
 * \f$ p^{n+1} \f$ is the current pressure, \f$ p^{n} \f$ is the previous pressure,
 * \f$ \Delta{t} \f$ is the current nodal time step, and \f$ alpha = \theta_1*\theta_2 \f$
 * denotes a scalar multiplier.
 *
 ******************************************************************************/
template<Plato::OrdinalType NumNodesPerCell,
         Plato::OrdinalType NumSpatialDims,
         typename ConfigT,
         typename CurPressT,
         typename PrevPressT,
         typename ResultT>
DEVICE_TYPE inline void
integrate_divergence_delta_pressure_gradient
(const Plato::OrdinalType & aCellOrdinal,
 const Plato::ScalarMultiVector & aTimeStep,
 const Plato::ScalarArray3DT<ConfigT> & aGradient,
 const Plato::ScalarVectorT<ConfigT> & aCellVolume,
 const Plato::ScalarMultiVectorT<CurPressT> & aCurPressGrad,
 const Plato::ScalarMultiVectorT<PrevPressT> & aPrevPressGrad,
 const Plato::ScalarMultiVectorT<ResultT> & aResult,
 Plato::Scalar aMultiplier = 1.0)
{
    for(Plato::OrdinalType tNode = 0; tNode < NumNodesPerCell; tNode++)
    {
        for(Plato::OrdinalType tDim = 0; tDim < NumSpatialDims; tDim++)
        {
            aResult(aCellOrdinal, tNode) += aMultiplier * aTimeStep(aCellOrdinal, tNode) *
                aCellVolume(aCellOrdinal) * aGradient(aCellOrdinal, tNode, tDim) *
                ( aCurPressGrad(aCellOrdinal, tDim) - aPrevPressGrad(aCellOrdinal, tDim) );
        }
    }
}

/***************************************************************************//**
 * \fn device_type void integrate_inertial_pressure_forces
 *
 * \brief Integrate inertial forces, which are defined as
 *
 * \f[
 *   \int_{\Omega}v^h \left(\frac{1}{\bar{c}^2}\right) \left( \bar{p}^{n+1} - \bar{p}^{n} \right)\, d\Omega
 * \f]
 *
 * where \f$ v^h \f$ is the test function, \f$ \bar{c} \f$ is the artificial
 * compressility, \f$ p^{n+1} \f$ is the current pressure, and \f$ p^{n} \f$ is
 * the previous pressure. Recall that the time step is move to the left hand
 * side, i.e. applied to the internal force vector.
 *
 ******************************************************************************/
template<Plato::OrdinalType NumNodesPerCell,
         typename ConfigT,
         typename PrevPressT,
         typename CurPressT,
         typename ResultT>
DEVICE_TYPE inline void
integrate_inertial_pressure_forces
(const Plato::OrdinalType & aCellOrdinal,
 const Plato::ScalarVector & aBasisFunctions,
 const Plato::ScalarVectorT<ConfigT> & aCellVolume,
 const Plato::ScalarVectorT<CurPressT> & aCurPress,
 const Plato::ScalarVectorT<PrevPressT> & aPrevPress,
 const Plato::ScalarMultiVector & aArtificialCompress,
 const Plato::ScalarMultiVectorT<ResultT> & aResult)
{
    for(Plato::OrdinalType tNode = 0; tNode < NumNodesPerCell; tNode++)
    {
        aResult(aCellOrdinal, tNode) += aCellVolume(aCellOrdinal) * aBasisFunctions(tNode) *
            ( static_cast<Plato::Scalar>(1.0) / aArtificialCompress(aCellOrdinal, tNode) ) *
            ( aCurPress(aCellOrdinal) - aPrevPress(aCellOrdinal) );
    }
}






template<typename PhysicsT, typename EvaluationT>
class MomentumSurfaceForces
{
private:
    static constexpr auto mNumSpatialDims       = PhysicsT::SimplexT::mNumSpatialDims;         /*!< number of spatial dimensions */
    static constexpr auto mNumNodesPerFace      = PhysicsT::SimplexT::mNumNodesPerFace;        /*!< number of nodes per face */
    static constexpr auto mNumVelDofsPerNode    = PhysicsT::SimplexT::mNumMomentumDofsPerNode; /*!< number of momentum dofs per node */
    static constexpr auto mNumSpatialDimsOnFace = PhysicsT::SimplexT::mNumSpatialDimsOnFace;   /*!< number of spatial dimensions on face */

    // forward automatic differentiation types
    using ResultT = typename EvaluationT::ResultScalarType;
    using ConfigT = typename EvaluationT::ConfigScalarType;
    using CurrentVelT = typename EvaluationT::CurrentMomentumScalarType;

    const std::string mEntitySetName; /*!< side set name */
    const Plato::SpatialDomain& mSpatialDomain; /*!< Plato spatial model */
    Plato::LinearTetCubRuleDegreeOne<mNumSpatialDims> mVolumeCubatureRule;  /*!< volume integration rule */
    Plato::LinearTetCubRuleDegreeOne<mNumSpatialDimsOnFace> mSurfaceCubatureRule; /*!< surface integration rule */

public:
    MomentumSurfaceForces
    (const Plato::SpatialDomain & aSpatialDomain,
     const std::string & aEntitySetName) :
         mEntitySetName(aEntitySetName),
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
        Plato::CalculateSurfaceArea<mNumSpatialDims> tCalculateSurfaceArea;
        Plato::NodeCoordinate<mNumSpatialDims> tCoords(&(mSpatialDomain.Mesh));
        Plato::CalculateSurfaceJacobians<mNumSpatialDims> tCalculateSurfaceJacobians;
        Plato::CreateFaceLocalNode2ElemLocalNodeIndexMap<mNumSpatialDims> tCreateFaceLocalNode2ElemLocalNodeIndexMap;
        Plato::InterpolateFromNodal<mNumSpatialDims, mNumVelDofsPerNode,   0/*offset*/, mNumSpatialDims> tIntrplVectorField;

        // get sideset faces
        auto tFaceLocalOrdinals = Plato::side_set_face_ordinals(mSpatialDomain.MeshSets, mEntitySetName);
        auto tNumFaces = tFaceLocalOrdinals.size();
        Plato::ScalarArray3DT<ConfigT> tJacobians("jacobian", tNumFaces, mNumSpatialDimsOnFace, mNumSpatialDims);
        auto tNumCells = mSpatialDomain.Mesh.nelems();
        Plato::ScalarMultiVectorT<CurrentVelT> tCurrentVelGP("current velocity", tNumCells, mNumVelDofsPerNode);

        // set input state worksets
        auto tConfigWS  = Plato::metadata<Plato::ScalarArray3DT<ConfigT>>(aWorkSets.get("configuration"));
        auto tCurrentVelWS = Plato::metadata<Plato::ScalarMultiVectorT<CurrentVelT>>(aWorkSets.get("current velocity"));

        // evaluate integral
        auto tVolumeBasisFunctions = mVolumeCubatureRule.getBasisFunctions();
        auto tSurfaceCubatureWeight = mSurfaceCubatureRule.getCubWeight();
        auto tSurfaceBasisFunctions = mSurfaceCubatureRule.getBasisFunctions();
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
              tCalculateSurfaceJacobians(tCellOrdinal, aFaceI, tLocalNodeOrd, tConfigWS, tJacobians);
              tCalculateSurfaceArea(aFaceI, tSurfaceCubatureWeight, tJacobians, tSurfaceAreaTimesCubWeight);

              // compute unit normal vector
              auto tElemFaceOrdinal = Plato::get_face_ordinal<mNumSpatialDims>(tCellOrdinal, tFaceOrdinal, tElem2Faces);
              auto tUnitNormalVec = Plato::unit_normal_vector(tCellOrdinal, tElemFaceOrdinal, tCoords);

              // project into aResult workset
              tIntrplVectorField(tCellOrdinal, tVolumeBasisFunctions, tCurrentVelWS, tCurrentVelGP);
              for( Plato::OrdinalType tNode = 0; tNode < mNumNodesPerFace; tNode++ )
              {
                  auto tLocalCellNode = tLocalNodeOrd[tNode];
                  for( Plato::OrdinalType tDim = 0; tDim < mNumSpatialDims; tDim++ )
                  {
                      aResult(tCellOrdinal, tLocalCellNode) += aMultiplier * tSurfaceBasisFunctions(tNode) *
                          tUnitNormalVec(tDim) * tCurrentVelGP(tCellOrdinal, tDim) * tSurfaceAreaTimesCubWeight;
                  }
              }
          }
        }, "calculate surface momentum integral");
    }
};
// class MomentumSurfaceForces







// todo: continuity equation
/***************************************************************************//**
 * \class PressureResidual
 *
 * \tparam PhysicsT    physics type
 * \tparam EvaluationT forward automatic differentiation evaluation type
 *
 * \brief Evaluate pressure equation residual, which is defined by
 *
 * \f[
 *   \mathcal{R}^n(v^h) = I^n(v^h) - F^n(v^h) - E^n(v^h) = 0.
 * \f]
 *
 * where
 *
 * Inertial Forces:
 *
 * \f[
 *   I^n(v^h) = \int_{\Omega}v^h & \left(\frac{1}{\bar{c}^2}\right)
 *     \left(\frac{\bar{p}^{n+1} - \bar{p}^{n}}{\Delta\bar{t}}\right)\, d\Omega
 * \f]
 *
 * Internal Forces:
 *
 * \f[
 *   F^n(v^h) = -\int_{\Omega}v^h\frac{\partial\bar{u}_i^{n}}{\partial\bar{x}_i}d\Omega
 *     + \theta_1\int_{\Omega}\frac{\partial v^h}{\partial\bar{x}_i}\left( \Delta{u}^{\ast}_i
 *     - \Delta\bar{t}\,\frac{\partial\bar{p}^{\,n+\theta_2}}{\partial\bar{x}_i} \right)d\Omega
 * \f]
 *
 * External Forces:
 *
 * \f[
 *   E^n(v^h) = -\theta_1\int_{\Gamma_u}v^h n_i \Delta{u}_i^n d\Gamma
 * \f]
 *
 * where
 *
 * \f[
 *   \frac{\partial\bar{p}^{\,n+\theta_2}}{\partial\bar{x}_i} =
 *     (1-\theta_2)\frac{\partial\bar{p}^{\,n}}{\partial\bar{x}_i} + \theta_2\frac{\partial\bar{p}^{\,n+1}}{\partial\bar{x}_i}
 *       = \frac{\partial\bar{p}^{\,n}}{\partial\bar{x}_i}+\theta_2\frac{\partial\Delta\bar{p}}{\partial\bar{x}_i}
 * \f]
 *
 ******************************************************************************/
template<typename PhysicsT, typename EvaluationT>
class PressureResidual : public Plato::Fluids::AbstractVectorFunction<PhysicsT, EvaluationT>
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
    using PrevVelT   = typename EvaluationT::PreviousMomentumScalarType;
    using PredVelT   = typename EvaluationT::MomentumPredictorScalarType;
    using CurPressT  = typename EvaluationT::CurrentMassScalarType;
    using PrevPressT = typename EvaluationT::PreviousMassScalarType;

    using CurPressGradT = typename Plato::Fluids::fad_type_t<typename PhysicsT::SimplexT, CurPressT, ConfigT>;
    using PrevPressGradT = typename Plato::Fluids::fad_type_t<typename PhysicsT::SimplexT, PrevPressT, ConfigT>;

    Plato::DataMap& mDataMap;                   /*!< output database */
    const Plato::SpatialDomain& mSpatialDomain; /*!< Plato spatial model */
    Plato::LinearTetCubRuleDegreeOne<mNumSpatialDims> mCubatureRule; /*!< integration rule */

    // artificial damping
    Plato::Scalar mPressDamping = 1.0;    /*!< artificial pressure damping */
    Plato::Scalar mMomentumDamping = 1.0; /*!< artificial momentum/velocity damping */

    // surface integral
    using MomentumForces = Plato::Fluids::MomentumSurfaceForces<PhysicsT, EvaluationT>;
    std::unordered_map<std::string, std::shared_ptr<MomentumForces>> mMomentumBCs;

public:
    PressureResidual
    (const Plato::SpatialDomain & aDomain,
     Plato::DataMap             & aDataMap,
     Teuchos::ParameterList     & aInputs) :
         mDataMap(aDataMap),
         mSpatialDomain(aDomain),
         mCubatureRule(Plato::LinearTetCubRuleDegreeOne<mNumSpatialDims>())
    {
        this->setSurfaceBoundaryIntegrals(aInputs);
        this->setAritificalPressureDamping(aInputs);
    }

    PressureResidual
    (const Plato::SpatialDomain & aDomain,
     Plato::DataMap             & aDataMap) :
         mDataMap(aDataMap),
         mSpatialDomain(aDomain),
         mCubatureRule(Plato::LinearTetCubRuleDegreeOne<mNumSpatialDims>())
    {
    }

    virtual ~PressureResidual(){}

    /***************************************************************************//**
     * \fn void evaluate
     *
     * \brief Evaluate internal pressure residual, which is defined by
     *
     * \f[
     *   \hat{R}^n(v^h) = I^n(v^h) - F^n(v^h) = 0.
     * \f]
     *
     * Inertial Forces:
     *
     * \f[
     *   I^n(v^h) = \int_{\Omega}v^h & \left(\frac{1}{\bar{c}^2}\right)
     *     \left(\frac{\bar{p}^{n+1} - \bar{p}^{n}}{\Delta\bar{t}}\right)\, d\Omega
     * \f]
     *
     * Internal Forces:
     *
     * \f[
     *   F^n(v^h) = -\int_{\Omega}v^h\frac{\partial\bar{u}_i^{n}}{\partial\bar{x}_i}d\Omega
     *     + \theta_1\int_{\Omega}\frac{\partial v^h}{\partial\bar{x}_i}\left( \Delta{u}^{\ast}_i
     *     - \Delta\bar{t}\,\frac{\partial\bar{p}^{\,n+\theta_2}}{\partial\bar{x}_i} \right)d\Omega
     * \f]
     *
     * where
     *
     * \f[
     *   \frac{\partial\bar{p}^{\,n+\theta_2}}{\partial\bar{x}_i} =
     *     (1-\theta_2)\frac{\partial\bar{p}^{\,n}}{\partial\bar{x}_i} + \theta_2\frac{\partial\bar{p}^{\,n+1}}{\partial\bar{x}_i}
     *       = \frac{\partial\bar{p}^{\,n}}{\partial\bar{x}_i}+\theta_2\frac{\partial\Delta\bar{p}}{\partial\bar{x}_i}
     * \f]
     *
     ******************************************************************************/
    void evaluate
    (const Plato::WorkSets & aWorkSets,
     Plato::ScalarMultiVectorT<ResultT> & aResultWS)
    const override
    {
        auto tNumCells = mSpatialDomain.numCells();
        if( tNumCells != static_cast<Plato::OrdinalType>(aResultWS.extent(0)) )
        {
            THROWERR(std::string("Number of elements mismatch. Spatial domain and output/result workset ")
                + "have different number of cells. " + "Spatial domain has '" + std::to_string(tNumCells)
                + "' elements and output workset has '" + std::to_string(aResultWS.extent(0)) + "' elements.")
        }

        // set local data
        Plato::ScalarVectorT<ConfigT> tCellVolume("cell weight", tNumCells);
        Plato::ScalarArray3DT<ConfigT> tGradient("cell gradient", tNumCells, mNumNodesPerCell, mNumSpatialDims);

        Plato::ScalarMultiVectorT<PrevVelT> tPrevVelGP("previous velocity", tNumCells, mNumSpatialDims);
        Plato::ScalarMultiVectorT<PredVelT> tPredVelGP("predicted velocity", tNumCells, mNumSpatialDims);
        Plato::ScalarMultiVectorT<ResultT> tRightHandSide("right hand side force", tNumCells, mNumPressDofsPerCell);
        Plato::ScalarMultiVectorT<CurPressGradT> tCurPressGradGP("current pressure gradient", tNumCells, mNumSpatialDims);
        Plato::ScalarMultiVectorT<PrevPressGradT> tPrevPressGradGP("previous pressure gradient", tNumCells, mNumSpatialDims);

        // set local functors
        Plato::ComputeGradientWorkset<mNumSpatialDims> tComputeGradient;
        Plato::InterpolateFromNodal<mNumSpatialDims, mNumVelDofsPerNode, 0/*offset*/, mNumSpatialDims> tIntrplVectorField;

        // set input state worksets
        auto tConfigWS    = Plato::metadata<Plato::ScalarArray3DT<ConfigT>>(aWorkSets.get("configuration"));
        auto tPrevVelWS   = Plato::metadata<Plato::ScalarMultiVectorT<PrevVelT>>(aWorkSets.get("previous velocity"));
        auto tPredVelWS   = Plato::metadata<Plato::ScalarMultiVectorT<PredVelT>>(aWorkSets.get("current predictor"));
        auto tCurPressWS  = Plato::metadata<Plato::ScalarMultiVectorT<CurPressT>>(aWorkSets.get("current pressure"));
        auto tPrevPressWS = Plato::metadata<Plato::ScalarMultiVectorT<PrevPressT>>(aWorkSets.get("previous pressure"));
        auto tCriticalTimeStep = Plato::metadata<Plato::ScalarVector>(aWorkSets.get("critical time step"));

        // transfer member data to device
        auto tPressDamping = mPressDamping;
        auto tMomentumDamping = mMomentumDamping;

        auto tCubWeight = mCubatureRule.getCubWeight();
        auto tBasisFunctions = mCubatureRule.getBasisFunctions();
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType &aCellOrdinal)
        {
            tComputeGradient(aCellOrdinal, tGradient, tConfigWS, tCellVolume);
            tCellVolume(aCellOrdinal) *= tCubWeight;

            // 1. add divergence of previous pressure gradient to residual, i.e. RHS += -1.0*\theta^{u}\Delta{t} L p^{n}
            Plato::Fluids::calculate_scalar_field_gradient<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tGradient, tPrevPressWS, tPrevPressGradGP);
            auto tMultiplier = tCriticalTimeStep(0) * tMomentumDamping;
            Plato::Fluids::integrate_laplacian_operator<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tGradient, tCellVolume, tPrevPressGradGP, tRightHandSide, -tMultiplier);

            // 2. add divergence of previous velocity to residual, RHS += Du_n
            tIntrplVectorField(aCellOrdinal, tBasisFunctions, tPrevVelWS, tPrevVelGP);
            Plato::Fluids::integrate_divergence_operator<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tBasisFunctions, tGradient, tCellVolume, tPrevVelGP, tRightHandSide);

            // 3. add divergence of delta predicted velocity to residual, RHS += D\Delta{\bar{u}}, where \Delta{\bar{u}} = \bar{u} - u_n
            tIntrplVectorField(aCellOrdinal, tBasisFunctions, tPredVelWS, tPredVelGP);
            Plato::Fluids::integrate_divergence_operator<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tBasisFunctions, tGradient, tCellVolume, tPredVelGP, tRightHandSide, tMomentumDamping);
            Plato::Fluids::integrate_divergence_operator<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tBasisFunctions, tGradient, tCellVolume, tPrevVelGP, tRightHandSide, -tMomentumDamping);

            // 4. apply \frac{1}{\Delta{t}} multiplier to right hand side, i.e. RHS = \frac{1}{\Delta{t}} * RHS
            tMultiplier = static_cast<Plato::Scalar>(1.0) / tCriticalTimeStep(0);
            Plato::blas1::scale<mNumPressDofsPerCell>(aCellOrdinal, tMultiplier, tRightHandSide);

            // 5. add divergence of current pressure gradient to residual, i.e. R += \theta^{p}\theta^{u} L p^{n+1}
            tMultiplier = tMomentumDamping * tPressDamping;
            Plato::Fluids::calculate_scalar_field_gradient<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tGradient, tCurPressWS, tCurPressGradGP);
            Plato::Fluids::integrate_laplacian_operator<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tGradient, tCellVolume, tCurPressGradGP, aResultWS, tMultiplier);

            // 6. add divergence of previous pressure gradient to residual, i.e. R -= \theta^{p}\theta^{u} L p^{n}
            Plato::Fluids::integrate_laplacian_operator<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tGradient, tCellVolume, tPrevPressGradGP, aResultWS, -tMultiplier);

            // 7. add right hand side force vector to residual, i.e. R -= RHS
            Plato::blas1::update<mNumPressDofsPerCell>(aCellOrdinal, -1.0, tRightHandSide, 1.0, aResultWS);
        }, "calculate continuity residual");
    }

    /***************************************************************************//**
     * \fn void evaluateBoundary
     *
     * \brief Evaluate momentum force increment applied on \f$ \Gamma_u \f$, which
     * is defined by
     *
     * \f[
     *   E^n(v^h) = -\theta_1\int_{\Gamma_u}v^h n_i \Delta{u}_i^n d\Gamma
     * \f]
     *
     * where \f$ \Delta{u}_i^n=\bar{u}_i^{n+1}-\bar{u}_i^{n} \f$
     *
     ******************************************************************************/
    void evaluateBoundary
    (const Plato::SpatialModel & aSpatialModel,
     const Plato::WorkSets & aWorkSets,
     Plato::ScalarMultiVectorT<ResultT> & aResultWS) const override
    {
        for(auto& tPair : mMomentumBCs)
        {
            tPair.second->operator()(aWorkSets, aResultWS);
        }
    }

    /***************************************************************************//**
     * \fn void evaluatePrescribed
     *
     * \brief Prescribed boundary forces are empty.
     *
     ******************************************************************************/
    void evaluatePrescribed
    (const Plato::SpatialModel & aSpatialModel,
     const Plato::WorkSets & aWorkSets,
     Plato::ScalarMultiVectorT<ResultT> & aResult) const override
    { return; }

private:
    void setAritificalPressureDamping(Teuchos::ParameterList& aInputs)
    {
        if(aInputs.isSublist("Time Integration"))
        {
            auto tTimeIntegration = aInputs.sublist("Time Integration");
            mPressDamping = tTimeIntegration.get<Plato::Scalar>("Pressure Damping", 1.0);
            mMomentumDamping = tTimeIntegration.get<Plato::Scalar>("Momentum Damping", 1.0);
        }
    }

    void setSurfaceBoundaryIntegrals(Teuchos::ParameterList& aInputs)
    {
        // the natural BCs are applied on the side sets where velocity BCs
        // are applied. therefore, the side sets corresponding to the velocity
        // BCs should be read by this function.
        std::unordered_map<std::string, std::vector<std::pair<Plato::OrdinalType, Plato::Scalar>>> tMap;
        if(aInputs.isSublist("Velocity Essential Boundary Conditions") == false)
        {
            THROWERR("'Velocity Essential Boundary Conditions' block must be defined for fluid flow problems.")
        }
        auto tSublist = aInputs.sublist("Velocity Essential Boundary Conditions");

        for (Teuchos::ParameterList::ConstIterator tItr = tSublist.begin(); tItr != tSublist.end(); ++tItr)
        {
            const Teuchos::ParameterEntry &tEntry = tSublist.entry(tItr);
            if (!tEntry.isList())
            {
                THROWERR(std::string("Error reading 'Velocity Essential Boundary Conditions' block: Expects a parameter ")
                    + "list input with information pertaining to the velocity boundary conditions .")
            }

            const std::string& tParamListName = tSublist.name(tItr);
            Teuchos::ParameterList & tParamList = tSublist.sublist(tParamListName);
            if (tParamList.isParameter("Sides") == false)
            {
                THROWERR(std::string("Keyword 'Sides' is not define in Parameter List '") + tParamListName + "'.")
            }
            const auto tEntitySetName = tParamList.get<std::string>("Sides");
            auto tMapItr = mMomentumBCs.find(tEntitySetName);
            if(tMapItr == mMomentumBCs.end())
            {
                mMomentumBCs[tEntitySetName] = std::make_shared<MomentumForces>(mSpatialDomain, tEntitySetName);
            }
        }
    }
};
// class PressureResidual





template<Plato::OrdinalType NumNodesPerCell,
         Plato::OrdinalType NumSpatialDims,
         typename ConfigT,
         typename FieldT,
         typename ResultT>
DEVICE_TYPE inline void
integrate_gradient_operator
(const Plato::OrdinalType & aCellOrdinal,
 const Plato::ScalarVector & aBasisFunctions,
 const Plato::ScalarVectorT<ConfigT> & aCellVolume,
 const Plato::ScalarMultiVectorT<FieldT> & aField,
 const Plato::ScalarMultiVectorT<ResultT> & aResult,
 Plato::Scalar aMultiplier = 1.0)
{
    for(Plato::OrdinalType tNode = 0; tNode < NumNodesPerCell; tNode++)
    {
        for(Plato::OrdinalType tDim = 0; tDim < NumSpatialDims; tDim++)
        {
            aResult(aCellOrdinal, tNode) += aMultiplier * aCellVolume(aCellOrdinal) * aBasisFunctions(tNode) * aField(aCellOrdinal, tDim);
        }
    }
}








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
    using ResidualEvalT      = typename Plato::Fluids::Evaluation<typename PhysicsT::SimplexT>::Residual;
    using GradConfigEvalT    = typename Plato::Fluids::Evaluation<typename PhysicsT::SimplexT>::GradConfig;
    using GradControlEvalT   = typename Plato::Fluids::Evaluation<typename PhysicsT::SimplexT>::GradControl;
    using GradCurVelEvalT    = typename Plato::Fluids::Evaluation<typename PhysicsT::SimplexT>::GradCurMomentum;
    using GradPrevVelEvalT   = typename Plato::Fluids::Evaluation<typename PhysicsT::SimplexT>::GradPrevMomentum;
    using GradCurTempEvalT   = typename Plato::Fluids::Evaluation<typename PhysicsT::SimplexT>::GradCurEnergy;
    using GradPrevTempEvalT  = typename Plato::Fluids::Evaluation<typename PhysicsT::SimplexT>::GradPrevEnergy;
    using GradCurPressEvalT  = typename Plato::Fluids::Evaluation<typename PhysicsT::SimplexT>::GradCurMass;
    using GradPrevPressEvalT = typename Plato::Fluids::Evaluation<typename PhysicsT::SimplexT>::GradPrevMass;
    using GradPredictorEvalT = typename Plato::Fluids::Evaluation<typename PhysicsT::SimplexT>::GradPredictor;

    // element residual vector function types
    using ResidualFuncT      = std::shared_ptr<Plato::Fluids::AbstractVectorFunction<PhysicsT, ResidualEvalT>>;
    using GradConfigFuncT    = std::shared_ptr<Plato::Fluids::AbstractVectorFunction<PhysicsT, GradConfigEvalT>>;
    using GradControlFuncT   = std::shared_ptr<Plato::Fluids::AbstractVectorFunction<PhysicsT, GradControlEvalT>>;
    using GradCurVelFuncT    = std::shared_ptr<Plato::Fluids::AbstractVectorFunction<PhysicsT, GradCurVelEvalT>>;
    using GradPrevVelFuncT   = std::shared_ptr<Plato::Fluids::AbstractVectorFunction<PhysicsT, GradPrevVelEvalT>>;
    using GradCurTempFuncT   = std::shared_ptr<Plato::Fluids::AbstractVectorFunction<PhysicsT, GradCurTempEvalT>>;
    using GradPrevTempFuncT  = std::shared_ptr<Plato::Fluids::AbstractVectorFunction<PhysicsT, GradPrevTempEvalT>>;
    using GradCurPressFuncT  = std::shared_ptr<Plato::Fluids::AbstractVectorFunction<PhysicsT, GradCurPressEvalT>>;
    using GradPrevPressFuncT = std::shared_ptr<Plato::Fluids::AbstractVectorFunction<PhysicsT, GradPrevPressEvalT>>;
    using GradPredictorFuncT = std::shared_ptr<Plato::Fluids::AbstractVectorFunction<PhysicsT, GradPredictorEvalT>>;

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
    Plato::VectorEntryOrdinal<mNumSpatialDims,mNumDofsPerNode> mStateOrdinalsMap;

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
        mLocalOrdinalMaps(aSpatialModel.Mesh),
        mStateOrdinalsMap(&aSpatialModel.Mesh)
    {
        this->initialize(aName, aDataMap, aInputs);
    }

    decltype(mNumSpatialDims) getNumSpatialDims() const
    {
        return mNumSpatialDims;
    }

    decltype(mNumDofsPerCell) getNumDofsPerCell() const
    {
        return mNumDofsPerCell;
    }

    decltype(mNumDofsPerNode) getNumDofsPerNode() const
    {
        return mNumDofsPerNode;
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
            Plato::Fluids::build_vector_function_worksets<ResidualEvalT>
                (tDomain, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            // evaluate internal forces
            auto tNumCells = tDomain.numCells();
            auto tName = tDomain.getDomainName();
            Plato::ScalarMultiVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells, mNumDofsPerCell);
            mResidualFuncs.at(tName)->evaluate(tInputWorkSets, tResultWS);
            Plato::assemble_residual<mNumNodesPerCell,mNumDofsPerNode>(tDomain, mStateOrdinalsMap, tResultWS, tReturnValue);
        }

        // evaluate boundary forces
        {
            Plato::WorkSets tInputWorkSets;
            auto tNumCells = mSpatialModel.Mesh.nelems();
            Plato::Fluids::build_vector_function_worksets<ResidualEvalT>
                (tNumCells, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            // evaluate prescribed forces
            Plato::ScalarMultiVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells, mNumDofsPerCell);
            mResidualFuncs.begin()->second->evaluatePrescribed(mSpatialModel, tInputWorkSets, tResultWS);
            Plato::assemble_residual<mNumNodesPerCell,mNumDofsPerNode>(tNumCells, mStateOrdinalsMap, tResultWS, tReturnValue);

            // evaluate balancing forces
            Plato::blas2::fill(0.0, tResultWS);
            mResidualFuncs.begin()->second->evaluateBoundary(mSpatialModel, tInputWorkSets, tResultWS);
            Plato::assemble_residual<mNumNodesPerCell,mNumDofsPerNode>(tNumCells, mStateOrdinalsMap, tResultWS, tReturnValue);
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
            Plato::Fluids::build_vector_function_worksets<GradConfigEvalT>
                (tDomain, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            // evaluate internal forces
            auto tName = tDomain.getDomainName();
            Plato::ScalarMultiVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells, mNumDofsPerCell);
            mGradConfigFuncs.at(tName)->evaluate(tInputWorkSets, tResultWS);
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumConfigDofsPerNode, mNumDofsPerNode> tJacEntryOrdinal(tJacobian, &tMesh);
            auto tJacobianMatrixEntries = tJacobian->entries();
            Plato::assemble_transpose_jacobian(tDomain, mNumDofsPerCell, mNumConfigDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobianMatrixEntries);
        }

        // evaluate boundary forces
        {
            Plato::WorkSets tInputWorkSets;
            auto tNumCells = mSpatialModel.Mesh.nelems();
            Plato::Fluids::build_vector_function_worksets<GradConfigEvalT>
                (tNumCells, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            // evaluate prescribed forces
            Plato::ScalarMultiVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells, mNumDofsPerCell);
            mGradConfigFuncs.begin()->second->evaluatePrescribed(mSpatialModel, tInputWorkSets, tResultWS);
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumConfigDofsPerNode, mNumDofsPerNode> tJacEntryOrdinal(tJacobian, &tMesh);
            auto tJacobianMatrixEntries = tJacobian->entries();
            Plato::assemble_transpose_jacobian(tNumCells, mNumDofsPerCell, mNumConfigDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobianMatrixEntries);

            // evaluate balancing forces
            Plato::blas2::fill(0.0, tResultWS);
            mGradConfigFuncs.begin()->second->evaluateBoundary(mSpatialModel, tInputWorkSets, tResultWS);
            Plato::assemble_transpose_jacobian(tNumCells, mNumDofsPerCell, mNumConfigDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobianMatrixEntries);
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
            Plato::Fluids::build_vector_function_worksets<GradControlEvalT>
                (tDomain, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            // evaluate internal forces
            auto tName = tDomain.getDomainName();
            Plato::ScalarMultiVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells, mNumDofsPerCell);
            mGradControlFuncs.at(tName)->evaluate(tInputWorkSets, tResultWS);
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumControlDofsPerNode, mNumDofsPerNode> tJacEntryOrdinal(tJacobian, &tMesh);
            auto tJacobianMatrixEntries = tJacobian->entries();
            Plato::assemble_transpose_jacobian(tDomain, mNumDofsPerCell, mNumControlDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobianMatrixEntries);
        }

        // evaluate boundary forces
        {
            Plato::WorkSets tInputWorkSets;
            auto tNumCells = mSpatialModel.Mesh.nelems();
            Plato::Fluids::build_vector_function_worksets<GradControlEvalT>
                (tNumCells, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            // evaluate prescribed forces
            Plato::ScalarMultiVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells, mNumDofsPerCell);
            mGradControlFuncs.begin()->second->evaluatePrescribed(mSpatialModel, tInputWorkSets, tResultWS);
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumControlDofsPerNode, mNumDofsPerNode> tJacEntryOrdinal(tJacobian, &tMesh);
            auto tJacobianMatrixEntries = tJacobian->entries();
            Plato::assemble_transpose_jacobian(tNumCells, mNumDofsPerCell, mNumControlDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobianMatrixEntries);

            // evaluate balancing forces
            Plato::blas2::fill(0.0, tResultWS);
            mGradControlFuncs.begin()->second->evaluateBoundary(mSpatialModel, tInputWorkSets, tResultWS);
            Plato::assemble_transpose_jacobian(tNumCells, mNumDofsPerCell, mNumControlDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobianMatrixEntries);
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
            Plato::Fluids::build_vector_function_worksets<GradPredictorEvalT>
                (tDomain, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            // evaluate internal forces
            auto tName = tDomain.getDomainName();
            Plato::ScalarMultiVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells, mNumDofsPerCell);
            mGradPredictorFuncs.at(tName)->evaluate(tInputWorkSets, tResultWS);
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumVelDofsPerNode, mNumDofsPerNode> tJacEntryOrdinal(tJacobian, &tMesh);
            auto tJacobianMatrixEntries = tJacobian->entries();
            Plato::assemble_transpose_jacobian(tDomain, mNumDofsPerCell, mNumVelDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobianMatrixEntries);
        }

        // evaluate boundary forces
        {
            Plato::WorkSets tInputWorkSets;
            auto tNumCells = mSpatialModel.Mesh.nelems();
            Plato::Fluids::build_vector_function_worksets<GradPredictorEvalT>
                (tNumCells, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            // evaluate prescribed forces
            Plato::ScalarMultiVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells, mNumDofsPerCell);
            mGradPredictorFuncs.begin()->second->evaluatePrescribed(mSpatialModel, tInputWorkSets, tResultWS);
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumVelDofsPerNode, mNumDofsPerNode> tJacEntryOrdinal(tJacobian, &tMesh);
            auto tJacobianMatrixEntries = tJacobian->entries();
            Plato::assemble_transpose_jacobian(tNumCells, mNumDofsPerCell, mNumVelDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobianMatrixEntries);

            // evaluate balancing forces
            Plato::blas2::fill(0.0, tResultWS);
            mGradPredictorFuncs.begin()->second->evaluateBoundary(mSpatialModel, tInputWorkSets, tResultWS);
            Plato::assemble_transpose_jacobian(tNumCells, mNumDofsPerCell, mNumVelDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobianMatrixEntries);
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
            Plato::Fluids::build_vector_function_worksets<GradPrevVelEvalT>
                (tDomain, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            // evaluate internal forces
            auto tName = tDomain.getDomainName();
            Plato::ScalarMultiVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells, mNumDofsPerCell);
            mGradPrevVelFuncs.at(tName)->evaluate(tInputWorkSets, tResultWS);
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumVelDofsPerNode, mNumDofsPerNode> tJacEntryOrdinal(tJacobian, &tMesh);
            auto tJacobianMatrixEntries = tJacobian->entries();
            Plato::assemble_transpose_jacobian(tDomain, mNumDofsPerCell, mNumVelDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobianMatrixEntries);
        }

        // evaluate boundary forces
        {
            Plato::WorkSets tInputWorkSets;
            auto tNumCells = mSpatialModel.Mesh.nelems();
            Plato::Fluids::build_vector_function_worksets<GradPrevVelEvalT>
                (tNumCells, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            // evaluate prescribed forces
            Plato::ScalarMultiVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells, mNumDofsPerCell);
            mGradPrevVelFuncs.begin()->second->evaluatePrescribed(mSpatialModel, tInputWorkSets, tResultWS);
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumVelDofsPerNode, mNumDofsPerNode> tJacEntryOrdinal(tJacobian, &tMesh);
            auto tJacobianMatrixEntries = tJacobian->entries();
            Plato::assemble_transpose_jacobian(tNumCells, mNumDofsPerCell, mNumVelDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobianMatrixEntries);

            // evaluate balancing forces
            Plato::blas2::fill(0.0, tResultWS);
            mGradPrevVelFuncs.begin()->second->evaluateBoundary(mSpatialModel, tInputWorkSets, tResultWS);
            Plato::assemble_transpose_jacobian(tNumCells, mNumDofsPerCell, mNumVelDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobianMatrixEntries);
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
            Plato::Fluids::build_vector_function_worksets<GradPrevPressEvalT>
                (tDomain, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            // evaluate internal forces
            auto tName = tDomain.getDomainName();
            Plato::ScalarMultiVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells, mNumDofsPerCell);
            mGradPrevPressFuncs.at(tName)->evaluate(tInputWorkSets, tResultWS);
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumPressDofsPerNode, mNumDofsPerNode> tJacEntryOrdinal(tJacobian, &tMesh);
            auto tJacobianMatrixEntries = tJacobian->entries();
            Plato::assemble_transpose_jacobian(tDomain, mNumDofsPerCell, mNumPressDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobianMatrixEntries);
        }

        // evaluate boundary forces
        {
            Plato::WorkSets tInputWorkSets;
            auto tNumCells = mSpatialModel.Mesh.nelems();
            Plato::Fluids::build_vector_function_worksets<GradPrevPressEvalT>
                (tNumCells, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            // evaluate prescribed forces
            Plato::ScalarMultiVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells, mNumDofsPerCell);
            mGradPrevPressFuncs.begin()->second->evaluatePrescribed(mSpatialModel, tInputWorkSets, tResultWS);
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumPressDofsPerNode, mNumDofsPerNode> tJacEntryOrdinal(tJacobian, &tMesh);
            auto tJacobianMatrixEntries = tJacobian->entries();
            Plato::assemble_transpose_jacobian(tNumCells, mNumDofsPerCell, mNumPressDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobianMatrixEntries);

            // evaluate balancing forces
            Plato::blas2::fill(0.0, tResultWS);
            mGradPrevPressFuncs.begin()->second->evaluateBoundary(mSpatialModel, tInputWorkSets, tResultWS);
            Plato::assemble_transpose_jacobian(tNumCells, mNumDofsPerCell, mNumPressDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobianMatrixEntries);
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
            Plato::Fluids::build_vector_function_worksets<GradPrevTempEvalT>
                (tDomain, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            // evaluate internal forces
            auto tName = tDomain.getDomainName();
            Plato::ScalarMultiVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells, mNumDofsPerCell);
            mGradPrevTempFuncs.at(tName)->evaluate(tInputWorkSets, tResultWS);
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumTempDofsPerNode, mNumDofsPerNode> tJacEntryOrdinal(tJacobian, &tMesh);
            auto tJacobianMatrixEntries = tJacobian->entries();
            Plato::assemble_transpose_jacobian(tDomain, mNumDofsPerCell, mNumTempDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobianMatrixEntries);
        }

        // evaluate boundary forces
        {
            Plato::WorkSets tInputWorkSets;
            auto tNumCells = mSpatialModel.Mesh.nelems();
            Plato::Fluids::build_vector_function_worksets<GradPrevTempEvalT>
                (tNumCells, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            // evaluate prescribed forces
            Plato::ScalarMultiVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells, mNumDofsPerCell);
            mGradPrevTempFuncs.begin()->second->evaluatePrescribed(mSpatialModel, tInputWorkSets, tResultWS);
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumTempDofsPerNode, mNumDofsPerNode> tJacEntryOrdinal(tJacobian, &tMesh);
            auto tJacobianMatrixEntries = tJacobian->entries();
            Plato::assemble_transpose_jacobian(tNumCells, mNumDofsPerCell, mNumTempDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobianMatrixEntries);

            // evaluate balancing forces
            Plato::blas2::fill(0.0, tResultWS);
            mGradPrevTempFuncs.begin()->second->evaluateBoundary(mSpatialModel, tInputWorkSets, tResultWS);
            Plato::assemble_transpose_jacobian(tNumCells, mNumDofsPerCell, mNumTempDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobianMatrixEntries);
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
            Plato::Fluids::build_vector_function_worksets<GradCurVelEvalT>
                (tDomain, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            // evaluate internal forces
            auto tName = tDomain.getDomainName();
            Plato::ScalarMultiVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells, mNumDofsPerCell);
            mGradCurVelFuncs.at(tName)->evaluate(tInputWorkSets, tResultWS);
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumVelDofsPerNode, mNumDofsPerNode> tJacEntryOrdinal(tJacobian, &tMesh);
            auto tJacobianMatrixEntries = tJacobian->entries();
            Plato::assemble_transpose_jacobian(tDomain, mNumDofsPerCell, mNumVelDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobianMatrixEntries);
        }

        // evaluate boundary forces
        {
            Plato::WorkSets tInputWorkSets;
            auto tNumCells = mSpatialModel.Mesh.nelems();
            Plato::Fluids::build_vector_function_worksets<GradCurVelEvalT>
                (tNumCells, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            // evaluate prescribed forces
            Plato::ScalarMultiVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells, mNumDofsPerCell);
            mGradCurVelFuncs.begin()->second->evaluatePrescribed(mSpatialModel, tInputWorkSets, tResultWS);
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumVelDofsPerNode, mNumDofsPerNode> tJacEntryOrdinal(tJacobian, &tMesh);
            auto tJacobianMatrixEntries = tJacobian->entries();
            Plato::assemble_transpose_jacobian(tNumCells, mNumDofsPerCell, mNumVelDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobianMatrixEntries);

            // evaluate balancing forces
            Plato::blas2::fill(0.0, tResultWS);
            mGradCurVelFuncs.begin()->second->evaluateBoundary(mSpatialModel, tInputWorkSets, tResultWS);
            Plato::assemble_transpose_jacobian(tNumCells, mNumDofsPerCell, mNumVelDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobianMatrixEntries);
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
            Plato::Fluids::build_vector_function_worksets<GradCurPressEvalT>
                (tDomain, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            // evaluate internal forces
            auto tName = tDomain.getDomainName();
            Plato::ScalarMultiVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells, mNumDofsPerCell);
            mGradCurPressFuncs.at(tName)->evaluate(tInputWorkSets, tResultWS);
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumPressDofsPerNode, mNumDofsPerNode> tJacEntryOrdinal(tJacobian, &tMesh);
            auto tJacobianMatrixEntires = tJacobian->entries();
            Plato::assemble_transpose_jacobian(tDomain, mNumDofsPerCell, mNumPressDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobianMatrixEntires);
        }

        // evaluate boundary forces
        {
            Plato::WorkSets tInputWorkSets;
            auto tNumCells = mSpatialModel.Mesh.nelems();
            Plato::Fluids::build_vector_function_worksets<GradCurPressEvalT>
                (tNumCells, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            // evaluate prescribed forces
            Plato::ScalarMultiVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells, mNumDofsPerCell);
            mGradCurPressFuncs.begin()->second->evaluatePrescribed(mSpatialModel, tInputWorkSets, tResultWS);
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumPressDofsPerNode, mNumDofsPerNode> tJacEntryOrdinal(tJacobian, &tMesh);
            auto tJacobianMatrixEntries = tJacobian->entries();
            Plato::assemble_transpose_jacobian(tNumCells, mNumDofsPerCell, mNumPressDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobianMatrixEntries);

            // evaluate balancing forces
            Plato::blas2::fill(0.0, tResultWS);
            mGradCurPressFuncs.begin()->second->evaluateBoundary(mSpatialModel, tInputWorkSets, tResultWS);
            Plato::assemble_transpose_jacobian(tNumCells, mNumDofsPerCell, mNumPressDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobianMatrixEntries);
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
            Plato::Fluids::build_vector_function_worksets<GradCurTempEvalT>
                (tDomain, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            // evaluate internal forces
            auto tName = tDomain.getDomainName();
            Plato::ScalarMultiVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells, mNumDofsPerCell);
            mGradCurTempFuncs.at(tName)->evaluate(tInputWorkSets, tResultWS);
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumTempDofsPerNode, mNumDofsPerNode> tJacEntryOrdinal(tJacobian, &tMesh);
            auto tJacobianMatrixEntries = tJacobian->entries();
            Plato::assemble_transpose_jacobian(tDomain, mNumDofsPerCell, mNumTempDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobianMatrixEntries);
        }

        // evaluate prescribed forces
        {
            Plato::WorkSets tInputWorkSets;
            auto tNumCells = mSpatialModel.Mesh.nelems();
            Plato::Fluids::build_vector_function_worksets<GradCurTempEvalT>
                (tNumCells, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            // evaluate boundary forces
            Plato::ScalarMultiVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells, mNumDofsPerCell);
            mGradCurTempFuncs.begin()->second->evaluatePrescribed(mSpatialModel, tInputWorkSets, tResultWS);
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumTempDofsPerNode, mNumDofsPerNode> tJacEntryOrdinal(tJacobian, &tMesh);
            auto tJacobianMatrixEntries = tJacobian->entries();
            Plato::assemble_transpose_jacobian(tNumCells, mNumDofsPerCell, mNumTempDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobianMatrixEntries);

            // evaluate balancing forces
            Plato::blas2::fill(0.0, tResultWS);
            mGradCurTempFuncs.begin()->second->evaluateBoundary(mSpatialModel, tInputWorkSets, tResultWS);
            Plato::assemble_transpose_jacobian(tNumCells, mNumDofsPerCell, mNumTempDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobianMatrixEntries);
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


template <typename PhysicsT, typename EvaluationT>
inline std::shared_ptr<Plato::Fluids::AbstractVectorFunction<PhysicsT, EvaluationT>>
temperature_residual
(const Plato::SpatialDomain & aDomain,
 Plato::DataMap & aDataMap,
 Teuchos::ParameterList & aInputs)
{
    if(aInputs.isSublist("Hyperbolic") == false)
    {
        THROWERR("'Hyperbolic' Parameter List is not defined.")
    }
    auto tHyperbolic = aInputs.sublist("Hyperbolic");

    auto tScenario = tHyperbolic.get<std::string>("Scenario","Analysis");
    auto tLowerScenario = Plato::tolower(tScenario);
    if( tLowerScenario == "density to" )
    {
        return ( std::make_shared<Plato::Fluids::SIMP::TemperatureResidual<PhysicsT, EvaluationT>>(aDomain, aDataMap, aInputs) );
    }
    else if( tLowerScenario == "analysis" || tLowerScenario == "levelset to" )
    {
        return ( std::make_shared<Plato::Fluids::TemperatureResidual<PhysicsT, EvaluationT>>(aDomain, aDataMap, aInputs) );
    }
    else
    {
        THROWERR(std::string("Scenario with tag '") + tScenario + "' is not supported. Options are 1) Analysis, 2) Density TO or 3) Levelset TO.")
    }
}

template <typename PhysicsT, typename EvaluationT>
inline std::shared_ptr<Plato::Fluids::AbstractVectorFunction<PhysicsT, EvaluationT>>
velocity_predictor_residual
(const Plato::SpatialDomain & aDomain,
 Plato::DataMap & aDataMap,
 Teuchos::ParameterList & aInputs)
{
    if(aInputs.isSublist("Hyperbolic") == false)
    {
        THROWERR("'Hyperbolic' Parameter List is not defined.")
    }
    auto tHyperbolic = aInputs.sublist("Hyperbolic");

    auto tScenario = tHyperbolic.get<std::string>("Scenario","Analysis");
    auto tLowerScenario = Plato::tolower(tScenario);
    if( tLowerScenario == "density to")
    {
        return ( std::make_shared<Plato::Fluids::Brinkman::VelocityPredictorResidual<PhysicsT, EvaluationT>>(aDomain, aDataMap, aInputs) );
    }
    else if( tLowerScenario == "analysis" || tLowerScenario == "levelset to" )
    {
        return ( std::make_shared<Plato::Fluids::VelocityPredictorResidual<PhysicsT, EvaluationT>>(aDomain, aDataMap, aInputs) );
    }
    else
    {
        THROWERR(std::string("Scenario with tag '") + tScenario + "' is not supported. Options are 1) Analysis, 2) Density TO or 3) Levelset TO.")
    }
}

struct FunctionFactory
{
public:
    template <typename PhysicsT, typename EvaluationT>
    std::shared_ptr<Plato::Fluids::AbstractVectorFunction<PhysicsT, EvaluationT>>
    createVectorFunction
    (const std::string          & aTag,
     const Plato::SpatialDomain & aDomain,
     Plato::DataMap             & aDataMap,
     Teuchos::ParameterList     & aInputs)
    {
        // TODO: explore function interface for constructor, similar to how it is done in the xml generator
        auto tLowerTag = Plato::tolower(aTag);
        if( tLowerTag == "pressure" )
        {
            return ( std::make_shared<Plato::Fluids::PressureResidual<PhysicsT, EvaluationT>>(aDomain, aDataMap, aInputs) );
        }
        else if( tLowerTag == "velocity corrector" )
        {
            return ( std::make_shared<Plato::Fluids::VelocityCorrectorResidual<PhysicsT, EvaluationT>>(aDomain, aDataMap, aInputs) );
        }
        else if( tLowerTag == "temperature" )
        {
            return ( Plato::Fluids::temperature_residual<PhysicsT, EvaluationT>(aDomain, aDataMap, aInputs) );
        }
        else if( tLowerTag == "velocity predictor" )
        {
            return ( Plato::Fluids::velocity_predictor_residual<PhysicsT, EvaluationT>(aDomain, aDataMap, aInputs) );
        }
        else
        {
            THROWERR(std::string("Vector function with tag '") + aTag + "' is not supported.")
        }
    }

    template <typename PhysicsT, typename EvaluationT>
    std::shared_ptr<Plato::Fluids::AbstractScalarFunction<PhysicsT, EvaluationT>>
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
            return ( std::make_shared<Plato::Fluids::AverageSurfacePressure<PhysicsT, EvaluationT>>
                (aName, aDomain, aDataMap, aInputs) );
        }
        else if( tCriterionLowerTag == "average surface temperature" )
        {
            return ( std::make_shared<Plato::Fluids::AverageSurfaceTemperature<PhysicsT, EvaluationT>>
                (aName, aDomain, aDataMap, aInputs) );
        }
        else if( tCriterionLowerTag == "internal dissipation energy" && tFlowLowerTag == "incompressible")
        {
            return ( std::make_shared<Plato::Fluids::InternalDissipationEnergy<PhysicsT, EvaluationT>>
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
    using ScalarFunctionType = std::shared_ptr<Plato::Fluids::CriterionBase>;

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
                std::make_shared<Plato::Fluids::ScalarFunction<PhysicsT>>
                    (aSpatialModel, aDataMap, aInputs, aName);
            return tCriterion;
        }
        /*else if(tLowerType == "weighted sum")
        {
            auto tCriterion =
                std::make_shared<Plato::Fluids::WeightedScalarFunction<PhysicsT>>
                    (aSpatialModel, aDataMap, aInputs, aName);
            return tCriterion;
        }
        else if(tLowerType == "least squares")
        {
            auto tCriterion =
                std::make_shared<Plato::Fluids::LeastSquaresScalarFunction<PhysicsT>>
                    (aSpatialModel, aDataMap, aInputs, aName);
            return tCriterion;
        }*/
        else
        {
            THROWERR(std::string("Scalar function in block '") + aName + "' with type '" + tType + "' is not supported.")
        }
     }
};


// todo: weighted scalar function
template<typename PhysicsT>
class WeightedScalarFunction : public Plato::Fluids::CriterionBase
{
private:
    // static metadata
    static constexpr auto mNumSpatialDims        = PhysicsT::SimplexT::mNumSpatialDims;         /*!< number of spatial dimensions */
    static constexpr auto mNumPressDofsPerNode   = PhysicsT::SimplexT::mNumMassDofsPerNode;     /*!< number of mass dofs per node */
    static constexpr auto mNumTempDofsPerNode    = PhysicsT::SimplexT::mNumEnergyDofsPerNode;   /*!< number of energy dofs per node */
    static constexpr auto mNumVelDofsPerNode     = PhysicsT::SimplexT::mNumMomentumDofsPerNode; /*!< number of momentum dofs per node */
    static constexpr auto mNumControlDofsPerNode = PhysicsT::SimplexT::mNumControlDofsPerNode;  /*!< number of design variables per node */

    // set local typenames
    using Criterion    = std::shared_ptr<Plato::Fluids::CriterionBase>;

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

        Plato::Fluids::CriterionFactory<PhysicsT> tFactory;
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
class LeastSquaresScalarFunction : public Plato::Fluids::CriterionBase
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
    using Criterion = std::shared_ptr<Plato::Fluids::CriterionBase>;

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

        Plato::Fluids::CriterionFactory<PhysicsT> tFactory;
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
// namespace Fluids


// todo: physics types
template<Plato::OrdinalType SpaceDim, Plato::OrdinalType NumControls = 1>
class MomentumConservation : public Plato::SimplexFluids<SpaceDim, NumControls>
{
public:
    typedef Plato::Fluids::FunctionFactory FunctionFactory;
    using SimplexT = Plato::SimplexFluids<SpaceDim, NumControls>;

    static constexpr Plato::OrdinalType mNumDofsPerNode = SimplexT::mNumMomentumDofsPerNode;
    static constexpr Plato::OrdinalType mNumDofsPerCell = SimplexT::mNumNodesPerCell * mNumDofsPerNode;
};

template<Plato::OrdinalType SpaceDim, Plato::OrdinalType NumControls = 1>
class MassConservation : public Plato::SimplexFluids<SpaceDim, NumControls>
{
public:
    typedef Plato::Fluids::FunctionFactory FunctionFactory;
    using SimplexT = Plato::SimplexFluids<SpaceDim, NumControls>;

    static constexpr Plato::OrdinalType mNumDofsPerNode = SimplexT::mNumMassDofsPerNode;
    static constexpr Plato::OrdinalType mNumDofsPerCell = SimplexT::mNumNodesPerCell * mNumDofsPerNode;
};

template<Plato::OrdinalType SpaceDim, Plato::OrdinalType NumControls = 1>
class EnergyConservation : public Plato::SimplexFluids<SpaceDim, NumControls>
{
public:
    typedef Plato::Fluids::FunctionFactory FunctionFactory;
    using SimplexT = Plato::SimplexFluids<SpaceDim, NumControls>;

    static constexpr Plato::OrdinalType mNumDofsPerNode = SimplexT::mNumEnergyDofsPerNode;
    static constexpr Plato::OrdinalType mNumDofsPerCell = SimplexT::mNumNodesPerCell * mNumDofsPerNode;
};

template<Plato::OrdinalType SpaceDim, Plato::OrdinalType NumControls = 1>
class IncompressibleFluids : public Plato::SimplexFluids<SpaceDim, NumControls>
{
public:
    static constexpr auto mNumSpatialDims = SpaceDim;

    typedef Plato::Fluids::FunctionFactory FunctionFactory;
    using SimplexT = typename Plato::SimplexFluids<SpaceDim, NumControls>;

    using MassPhysicsT     = typename Plato::MassConservation<SpaceDim, NumControls>;
    using EnergyPhysicsT   = typename Plato::EnergyConservation<SpaceDim, NumControls>;
    using MomentumPhysicsT = typename Plato::MomentumConservation<SpaceDim, NumControls>;
};




// todo: unit test inline functions
namespace cbs
{

template<Plato::OrdinalType NumSpatialDims,
         Plato::OrdinalType NumNodesPerCell>
inline Plato::ScalarVector
calculate_element_characteristic_sizes
(const Plato::SpatialModel & aSpatialModel)
{
    auto tCoords = aSpatialModel.Mesh.coords();
    auto tCells2Nodes = aSpatialModel.Mesh.ask_elem_verts();

    Plato::OrdinalType tNumCells = aSpatialModel.Mesh.nelems();
    Plato::OrdinalType tNumNodes = aSpatialModel.Mesh.nverts();
    Plato::ScalarVector tElemCharSize("element characteristic size", tNumNodes);
    Plato::blas1::fill(std::numeric_limits<Plato::Scalar>::max(), tElemCharSize);

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        auto tElemSize = Plato::omega_h::calculate_element_size<NumSpatialDims,NumNodesPerCell>(aCellOrdinal, tCells2Nodes, tCoords);
        for(Plato::OrdinalType tNode = 0; tNode < NumNodesPerCell; tNode++)
        {
            auto tVertexIndex = tCells2Nodes[aCellOrdinal*NumNodesPerCell + tNode];
            tElemCharSize(tVertexIndex) = tElemSize <= tElemCharSize(tVertexIndex) ? tElemSize : tElemCharSize(tVertexIndex);
        }
    },"calculate characteristic element size");

    return tElemCharSize;
}

template<Plato::OrdinalType NodesPerCell>
Plato::ScalarVector
calculate_convective_velocity_magnitude
(const Plato::SpatialModel & aSpatialModel,
 const Plato::ScalarVector & aVelocityField)
{
    auto tCell2Node = aSpatialModel.Mesh.ask_elem_verts();
    Plato::OrdinalType tSpaceDim = aSpatialModel.Mesh.dim();
    Plato::OrdinalType tNumCells = aSpatialModel.Mesh.nelems();
    Plato::OrdinalType tNumNodes = aSpatialModel.Mesh.nverts();

    Plato::ScalarVector tConvectiveVelocity("convective velocity", tNumNodes);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCell)
    {
        for(Plato::OrdinalType tNode = 0; tNode < NodesPerCell; tNode++)
        {
            Plato::Scalar tSum = 0.0;
            Plato::OrdinalType tVertexIndex = tCell2Node[aCell*NodesPerCell + tNode];
            for(Plato::OrdinalType tDim = 0; tDim < tSpaceDim; tDim++)
            {
                auto tDofIndex = tVertexIndex * tSpaceDim + tDim;
                tSum += aVelocityField(tDofIndex) * aVelocityField(tDofIndex);
            }
            auto tMyValue = sqrt(tSum);
            tConvectiveVelocity(tVertexIndex) =
                tMyValue >= tConvectiveVelocity(tVertexIndex) ? tMyValue : tConvectiveVelocity(tVertexIndex);
        }
    }, "calculate_convective_velocity_magnitude");

    return tConvectiveVelocity;
}

template<Plato::OrdinalType NodesPerCell>
Plato::ScalarVector
calculate_diffusive_velocity_magnitude
(const Plato::SpatialModel & aSpatialModel,
 const Plato::Scalar & aReynoldsNum,
 const Plato::ScalarVector & aCharElemSize)
{
    auto tCell2Node = aSpatialModel.Mesh.ask_elem_verts();
    Plato::OrdinalType tNumCells = aSpatialModel.Mesh.nelems();
    Plato::OrdinalType tNumNodes = aSpatialModel.Mesh.nverts();

    Plato::ScalarVector tDiffusiveVelocity("diffusive velocity", tNumNodes);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        for(decltype(tNumNodes) tNode = 0; tNode < NodesPerCell; tNode++)
        {
            Plato::OrdinalType tVertexIndex = tCell2Node[aCellOrdinal*NodesPerCell + tNode];
            auto tMyValue = static_cast<Plato::Scalar>(1.0) / (aCharElemSize(tVertexIndex) * aReynoldsNum);
            tDiffusiveVelocity(tVertexIndex) =
                tMyValue >= tDiffusiveVelocity(tVertexIndex) ? tMyValue : tDiffusiveVelocity(tVertexIndex);
        }
    }, "calculate_diffusive_velocity_magnitude");

    return tDiffusiveVelocity;
}

template<Plato::OrdinalType NodesPerCell>
Plato::ScalarVector
calculate_thermal_velocity_magnitude
(const Plato::SpatialModel & aSpatialModel,
 const Plato::Scalar & aPrandtlNum,
 const Plato::Scalar & aReynoldsNum,
 const Plato::ScalarVector & aCharElemSize)
{
    auto tCell2Node = aSpatialModel.Mesh.ask_elem_verts();
    Plato::OrdinalType tNumCells = aSpatialModel.Mesh.nelems();
    Plato::OrdinalType tNumNodes = aSpatialModel.Mesh.nverts();

    Plato::ScalarVector tThermalVelocity("thermal velocity", tNumNodes);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        for(decltype(tNumNodes) tNode = 0; tNode < NodesPerCell; tNode++)
        {
            Plato::OrdinalType tVertexIndex = tCell2Node[aCellOrdinal*NodesPerCell + tNode];
            auto tMyValue = static_cast<Plato::Scalar>(1.0) / (aCharElemSize(tVertexIndex) * aReynoldsNum * aPrandtlNum);
            tThermalVelocity(tVertexIndex) =
                tMyValue >= tThermalVelocity(tVertexIndex) ? tMyValue : tThermalVelocity(tVertexIndex);
        }
    }, "calculate_thermal_velocity_magnitude");

    return tThermalVelocity;
}

/***************************************************************************//**
 *  \brief Calculate artificial compressibility for incompressible flow problems.
 *  The artificial compressibility is computed as follows:
 *  \f$ \beta=\max(\epsilon,u_{convective},u_{diffusive},u_{thermal}) \f$,
 *  where
 *  \f$ u_{convective} = \sqrt(u_iu_i),\quad\in\{1,\dots,\mbox{dim}\} \f$
 *  \f$ u_{diffusive}  = \frac{1.0}{h_e\mbox{Re}} \f$
 *  \f$ u_{thermal}    = \frac{1.0}{h_e\mbox{Re}\mbox{Pr}} \f$
 *  Here, $h_e$ is the $e$-th element characteristic length, Re is the Reynolds number,
 *  Pr is the Prandtl number
 *
 *  \param aStates                [in] metadata structure with current set of primal states
 *  \param aCritialCompresibility [in] artificial compressibility lower bound
 *
 ******************************************************************************/
inline Plato::ScalarVector
calculate_artificial_compressibility
(const Plato::ScalarVector & aConvectiveVelocity,
 const Plato::ScalarVector & aDiffusiveVelocity,
 const Plato::ScalarVector & aThermalVelocity,
 Plato::Scalar aCritialCompresibility = 0.5)
{
    Plato::OrdinalType tNumNodes = aThermalVelocity.size();
    Plato::ScalarVector tArtificialCompressibility("artificial compressibility", tNumNodes);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumNodes), LAMBDA_EXPRESSION(const Plato::OrdinalType & aNode)
    {
        auto tMyArtificialCompressibility = (aConvectiveVelocity(aNode) >= aDiffusiveVelocity(aNode))
            && (aConvectiveVelocity(aNode) >= aThermalVelocity(aNode))
            && (aConvectiveVelocity(aNode) >= aCritialCompresibility) ?
                aConvectiveVelocity(aNode) : aCritialCompresibility;

        tMyArtificialCompressibility = (aDiffusiveVelocity(aNode) >= aConvectiveVelocity(aNode) )
            && (aDiffusiveVelocity(aNode) >= aThermalVelocity(aNode))
            && (aDiffusiveVelocity(aNode) >= aCritialCompresibility) ?
                aDiffusiveVelocity(aNode) : tMyArtificialCompressibility;

        tMyArtificialCompressibility = (aThermalVelocity(aNode) >= aConvectiveVelocity(aNode) )
            && (aThermalVelocity(aNode) >= aDiffusiveVelocity(aNode))
            && (aThermalVelocity(aNode) >= aCritialCompresibility) ?
                aThermalVelocity(aNode) : tMyArtificialCompressibility;

        tArtificialCompressibility(aNode) = tMyArtificialCompressibility;
    }, "calculate_artificial_compressibility");

    return tArtificialCompressibility;
}

inline Plato::Scalar
calculate_critical_convective_time_step
(const Plato::SpatialModel & aSpatialModel,
 const Plato::ScalarVector & aElemCharSize,
 const Plato::ScalarVector & aVelocityField,
 Plato::Scalar aSafetyFactor = 0.7)
{
    auto tNorm = Plato::blas1::norm(aVelocityField);
    if(tNorm <= std::numeric_limits<Plato::Scalar>::min())
    {
        return std::numeric_limits<Plato::Scalar>::max();
    }

    auto tNumNodes = aSpatialModel.Mesh.nverts();
    Plato::ScalarVector tLocalTimeStep("time step", tNumNodes);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumNodes), LAMBDA_EXPRESSION(const Plato::OrdinalType & aNodeOrdinal)
    {
        tLocalTimeStep(aNodeOrdinal) = aSafetyFactor * ( aElemCharSize(aNodeOrdinal) / aVelocityField(aNodeOrdinal) );
    }, "calculate local critical time step");

    Plato::Scalar tMinValue = 0;
    Plato::blas1::min(tLocalTimeStep, tMinValue);
    return tMinValue;
}

inline Plato::Scalar
calculate_critical_thermal_time_step
(const Plato::ScalarVector & aElemCharSize,
 const Plato::Scalar & aPrNum,
 Plato::Scalar aSafetyFactor = 0.7)
{
    Plato::Scalar tMinValue = 0;
    Plato::blas1::min(aElemCharSize, tMinValue);
    auto tCriticalStep = tMinValue * tMinValue *  static_cast<Plato::Scalar>(2) * aPrNum;
    return tCriticalStep;
}

inline Plato::Scalar
calculate_critical_diffusive_time_step
(const Plato::SpatialModel & aSpatialModel,
 const Plato::ScalarVector & aElemCharSize,
 const Plato::Scalar & aReynoldsNumber,
 const Plato::Scalar & aPrandtlNumber,
 Plato::Scalar aSafetyFactor = 0.7)
{
    auto tNumNodes = aSpatialModel.Mesh.nverts();
    Plato::ScalarVector tLocalTimeStep("time step", tNumNodes);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumNodes), LAMBDA_EXPRESSION(const Plato::OrdinalType & aNodeOrdinal)
    {
        auto tOptionOne = static_cast<Plato::Scalar>(0.5) * aElemCharSize(aNodeOrdinal) * aElemCharSize(aNodeOrdinal) * aReynoldsNumber;
        auto tOptionTwo = static_cast<Plato::Scalar>(0.5) * aElemCharSize(aNodeOrdinal) * aElemCharSize(aNodeOrdinal) * aReynoldsNumber * aPrandtlNumber;
        tLocalTimeStep(aNodeOrdinal) = tOptionOne < tOptionTwo ? tOptionOne : tOptionTwo;
    }, "calculate local critical time step");

    Plato::Scalar tMinValue = 0;
    Plato::blas1::min(tLocalTimeStep, tMinValue);
    return tMinValue;
}

inline void
enforce_boundary_condition
(const Plato::LocalOrdinalVector & aBcDofs,
 const Plato::ScalarVector       & aBcValues,
 const Plato::ScalarVector       & aState)
{
    auto tLength = aBcValues.size();
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tLength), LAMBDA_EXPRESSION(const Plato::OrdinalType & aOrdinal)
    {
        auto tDOF = aBcDofs(aOrdinal);
        aState(tDOF) = aBcValues(aOrdinal);
    }, "enforce boundary condition");
}

template
<Plato::OrdinalType DofsPerNode>
inline Plato::ScalarVector
calculate_field_misfit
(const Plato::OrdinalType & aNumNodes,
 const Plato::ScalarVector& aFieldOne,
 const Plato::ScalarVector& aFieldTwo)
{
    Plato::ScalarVector tResidual("pressure residual", aNumNodes * DofsPerNode);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, aNumNodes), LAMBDA_EXPRESSION(const Plato::OrdinalType & aNode)
    {
        for(Plato::OrdinalType tDof = 0; tDof < DofsPerNode; tDof++)
        {
            Plato::OrdinalType tLocalDof = aNode * DofsPerNode + tDof;
            tResidual(tLocalDof) = aFieldOne(tLocalDof) - aFieldTwo(tLocalDof);
        }
    }, "calculate stopping criterion");

    return tResidual;
}

template
<Plato::OrdinalType DofsPerNode>
inline Plato::Scalar
calculate_misfit_euclidean_norm
(const Plato::OrdinalType aNumNodes,
 const Plato::ScalarVector& aFieldOne,
 const Plato::ScalarVector& aFieldTwo)
{
    auto tResidual = Plato::cbs::calculate_field_misfit<DofsPerNode>(aNumNodes, aFieldOne, aFieldTwo);
    auto tValue = Plato::blas1::norm(tResidual);
    return tValue;
}

template
<Plato::OrdinalType DofsPerNode>
inline Plato::Scalar
calculate_misfit_inf_norm
(const Plato::OrdinalType aNumNodes,
 const Plato::ScalarVector& aFieldOne,
 const Plato::ScalarVector& aFieldTwo)
{
    auto tMyResidual = Plato::cbs::calculate_field_misfit<DofsPerNode>(aNumNodes, aFieldOne, aFieldTwo);

    Plato::Scalar tOutput = 0.0;
    Plato::blas1::abs(tMyResidual);
    Plato::blas1::max(tMyResidual, tOutput);

    return tOutput;
}

}
// namespace cbs

template<Plato::OrdinalType NumDofsPerNode>
inline void apply_constraints
(const Plato::LocalOrdinalVector & aBcDofs,
 const Plato::ScalarVector & aBcValues,
 const Teuchos::RCP<Plato::CrsMatrixType> & aMatrix,
 Plato::ScalarVector & aRhs,
 Plato::Scalar aScale = 1.0)
{
    if(aMatrix->isBlockMatrix())
    {
        Plato::applyBlockConstraints<NumDofsPerNode>(aMatrix, aRhs, aBcDofs, aBcValues, aScale);
    }
    else
    {
        Plato::applyConstraints<NumDofsPerNode>(aMatrix, aRhs, aBcDofs, aBcValues, aScale);
    }
}

inline void set_dofs_values
(const Plato::LocalOrdinalVector & aBcDofs,
       Plato::ScalarVector & aOutput,
       Plato::Scalar aValue = 0.0)
{
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, aBcDofs.size()), LAMBDA_EXPRESSION(const Plato::OrdinalType & aOrdinal)
    {
        aOutput(aBcDofs(aOrdinal)) = aValue;
    }, "set values at bc dofs to zero");
}

inline void open_text_file
(const std::string & aFileName,
 std::ofstream & aTextFile,
 bool aPrint = true)
{
    if (aPrint == false)
    {
        return;
    }
    aTextFile.open(aFileName);
}

inline void close_text_file
(std::ofstream & aTextFile,
 bool aPrint = true)
{
    if (aPrint == false)
    {
        return;
    }
    aTextFile.close();
}

inline void append_text_to_file
(const std::stringstream & aMsg,
 std::ofstream & aTextFile,
 bool aPrint = true)
{
    if (aPrint == false)
    {
        return;
    }
    aTextFile << aMsg.str().c_str() << std::flush;
}

namespace Fluids
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
class QuasiImplicit : public Plato::Fluids::AbstractProblem
{
private:
    static constexpr auto mNumSpatialDims      = PhysicsT::mNumSpatialDims;         /*!< number of spatial dimensions */
    static constexpr auto mNumNodesPerCell     = PhysicsT::mNumNodesPerCell;        /*!< number of nodes per cell */
    static constexpr auto mNumVelDofsPerNode   = PhysicsT::mNumMomentumDofsPerNode; /*!< number of momentum dofs per node */
    static constexpr auto mNumTempDofsPerNode  = PhysicsT::mNumEnergyDofsPerNode;   /*!< number of energy dofs per node */
    static constexpr auto mNumPressDofsPerNode = PhysicsT::mNumMassDofsPerNode;     /*!< number of mass dofs per node */

    Plato::Comm::Machine& mMachine; /*!< parallel communication interface */
    const Teuchos::ParameterList& mInputs; /*!< plato problem inputs */

    Plato::DataMap mDataMap; /*!< static output fields metadata interface */
    Plato::SpatialModel mSpatialModel; /*!< SpatialModel instance contains the mesh, meshsets, domains, etc. */

    bool mPrintDiagnostics = true;
    bool mCalculateHeatTransfer = false;

    std::ofstream mDiagnostics; /*!< output diagnostics */

    Plato::Scalar mPrandtlNumber = 1.0;
    Plato::Scalar mPressureTolerance = 1e-4;
    Plato::Scalar mPredictorTolerance = 1e-4;
    Plato::Scalar mCorrectorTolerance = 1e-4;
    Plato::Scalar mTemperatureTolerance = 1e-2;
    Plato::Scalar mSteadyStateTolerance = 1e-5;
    Plato::Scalar mTimeStepSafetyFactor = 0.9; /*!< safety factor applied to stable time step */

    Plato::OrdinalType mOutputFrequency = 1e6; 
    Plato::OrdinalType mMaxPressureIterations = 5; /*!< maximum number of pressure solver iterations */
    Plato::OrdinalType mMaxPredictorIterations = 5; /*!< maximum number of predictor solver iterations */
    Plato::OrdinalType mMaxCorrectorIterations = 5; /*!< maximum number of corrector solver iterations */
    Plato::OrdinalType mMaxTemperatureIterations = 5; /*!< maximum number of temperature solver iterations */
    Plato::OrdinalType mMaxSteadyStateIterations = 2000; /*!< maximum number of steady state iterations */

    Plato::ScalarMultiVector mPressure;
    Plato::ScalarMultiVector mVelocity;
    Plato::ScalarMultiVector mPredictor;
    Plato::ScalarMultiVector mTemperature;

    Plato::Fluids::VectorFunction<typename PhysicsT::MassPhysicsT>     mPressureResidual;
    Plato::Fluids::VectorFunction<typename PhysicsT::MomentumPhysicsT> mPredictorResidual;
    Plato::Fluids::VectorFunction<typename PhysicsT::MomentumPhysicsT> mCorrectorResidual;
    // Using pointer since default VectorFunction constructor allocations are not permitted.
    // Temperature VectorFunction allocation is optional since heat transfer calculations are optional
    std::shared_ptr<Plato::Fluids::VectorFunction<typename PhysicsT::EnergyPhysicsT>> mTemperatureResidual;

    using Criterion = std::shared_ptr<Plato::Fluids::CriterionBase>;
    using Criteria  = std::unordered_map<std::string, Criterion>;
    Criteria mCriteria;

    using MassConservationT     = typename Plato::MassConservation<PhysicsT::mNumSpatialDims, PhysicsT::mNumControlDofsPerNode>;
    using EnergyConservationT   = typename Plato::EnergyConservation<PhysicsT::mNumSpatialDims, PhysicsT::mNumControlDofsPerNode>;
    using MomentumConservationT = typename Plato::MomentumConservation<PhysicsT::mNumSpatialDims, PhysicsT::mNumControlDofsPerNode>;
    Plato::EssentialBCs<MassConservationT>     mPressureEssentialBCs;
    Plato::EssentialBCs<MomentumConservationT> mVelocityEssentialBCs;
    Plato::EssentialBCs<EnergyConservationT>   mTemperatureEssentialBCs;

public:
    QuasiImplicit
    (Omega_h::Mesh          & aMesh,
     Omega_h::MeshSets      & aMeshSets,
     Teuchos::ParameterList & aInputs,
     Plato::Comm::Machine   & aMachine) :
         mMachine(aMachine),
         mInputs(aInputs),
         mSpatialModel(aMesh, aMeshSets, aInputs),
         mPressureResidual("Pressure", mSpatialModel, mDataMap, aInputs),
         mCorrectorResidual("Velocity Corrector", mSpatialModel, mDataMap, aInputs),
         mPredictorResidual("Velocity Predictor", mSpatialModel, mDataMap, aInputs),
         mPressureEssentialBCs(aInputs.sublist("Pressure Essential Boundary Conditions",false),aMeshSets),
         mVelocityEssentialBCs(aInputs.sublist("Velocity Essential Boundary Conditions",false),aMeshSets),
         mTemperatureEssentialBCs(aInputs.sublist("Temperature Essential Boundary Conditions",false),aMeshSets)
    {
        this->initialize(aInputs);
    }

    const decltype(mDataMap)& getDataMap() const
    {
        return mDataMap;
    }

    ~QuasiImplicit()
    {
        if(Plato::Comm::rank(mMachine) == 0)
        {
            Plato::close_text_file(mDiagnostics, mPrintDiagnostics);
        }
    }

    void output(std::string aFilePath = "output")
    {
        auto tMesh = mSpatialModel.Mesh;
        auto tWriter = Omega_h::vtk::Writer(aFilePath.c_str(), &tMesh, mNumSpatialDims);

        constexpr auto tStride = 0;
        constexpr auto tCurrentTimeStep = 1;
        const auto tNumNodes = tMesh.nverts();

        auto tPressSubView = Kokkos::subview(mPressure, 0, Kokkos::ALL());
        Omega_h::Write<Omega_h::Real> tPressure(tPressSubView.size(), "Pressure");
        Plato::copy<mNumPressDofsPerNode, mNumPressDofsPerNode>(tStride, tNumNodes, tPressSubView, tPressure);
        tMesh.add_tag(Omega_h::VERT, "Pressure", mNumPressDofsPerNode, Omega_h::Reals(tPressure));

        auto tVelSubView = Kokkos::subview(mVelocity, tCurrentTimeStep, Kokkos::ALL());
        Omega_h::Write<Omega_h::Real> tVelocity(tVelSubView.size(), "Velocity");
        Plato::copy<mNumVelDofsPerNode, mNumVelDofsPerNode>(tStride, tNumNodes, tVelSubView, tVelocity);
        tMesh.add_tag(Omega_h::VERT, "Velocity", mNumVelDofsPerNode, Omega_h::Reals(tVelocity));

        if(mCalculateHeatTransfer)
        {
            auto tTempSubView = Kokkos::subview(mTemperature, tCurrentTimeStep, Kokkos::ALL());
            Omega_h::Write<Omega_h::Real> tTemperature(tTempSubView.size(), "Temperature");
            Plato::copy<mNumTempDofsPerNode, mNumTempDofsPerNode>(tStride, tNumNodes, tTempSubView, tTemperature);
            tMesh.add_tag(Omega_h::VERT, "Temperature", mNumTempDofsPerNode, Omega_h::Reals(tTemperature));
        }

        auto tTags = Omega_h::vtk::get_all_vtk_tags(&tMesh, mNumSpatialDims);
        auto tTime = static_cast<Plato::Scalar>(tCurrentTimeStep);
        tWriter.write(tCurrentTimeStep, tTime, tTags);
    }

    Plato::Solutions solution
    (const Plato::ScalarVector& aControl)
    {
        this->checkProblemSetup();

        Plato::Primal tPrimal;
        this->setInitialConditions(tPrimal);
        this->calculateElemCharacteristicSize(tPrimal);

        for(Plato::OrdinalType tIteration = 0; tIteration < mMaxSteadyStateIterations; tIteration++)
        {
            tPrimal.scalar("iteration", tIteration+1);

            this->setPrimalStates(tPrimal);
            this->calculateCriticalTimeStep(tPrimal);

            this->printIteration(tPrimal);
            this->updatePredictor(aControl, tPrimal);
            this->updatePressure(aControl, tPrimal);
            this->updateCorrector(aControl, tPrimal);

            if(mCalculateHeatTransfer)
            {
                this->updateTemperature(aControl, tPrimal);
            }

	    if(tIteration == mOutputFrequency)
            {
                this->output();
            }

            this->updatePreviousStates(tPrimal);
            if(this->checkStoppingCriteria(tPrimal))
            {
                break;
            }
        }

        auto tSolution = this->setOutputSolution();
        return tSolution;
    }

    Plato::Scalar criterionValue
    (const Plato::ScalarVector & aControl,
     const std::string         & aName)
    {
        auto tItr = mCriteria.find(aName);
        if (tItr == mCriteria.end())
        {
            Plato::Scalar tOutput(0);
            Plato::Primal tPrimal;

            constexpr Plato::OrdinalType tSteadyStateStep = 1;
            auto tPressure = Kokkos::subview(mPressure, tSteadyStateStep, Kokkos::ALL());
            auto tVelocity = Kokkos::subview(mVelocity, tSteadyStateStep, Kokkos::ALL());
            auto tTemperature = Kokkos::subview(mTemperature, tSteadyStateStep, Kokkos::ALL());
            tPrimal.vector("current pressure", tPressure);
            tPrimal.vector("current velocity", tVelocity);
            tPrimal.vector("current temperature", tTemperature);
            tOutput += tItr->second->value(aControl, tPrimal);
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
            Plato::Dual tDual;
            tDual.scalar("step", 1.0);
            this->setDualVariables(tDual);

            Plato::Primal tPrimal;
            tPrimal.scalar("step", 1.0);
            this->setPrimalStates(tPrimal);

            this->calculateElemCharacteristicSize(tPrimal);
            this->calculateCriticalTimeStep(tPrimal);

            this->calculateCorrectorAdjoint(aName, aControl, tPrimal, tDual);
            this->calculateTemperatureAdjoint(aName, aControl, tPrimal, tDual);
            this->calculatePressureAdjoint(aName, aControl, tPrimal, tDual);
            this->calculatePredictorAdjoint(aControl, tPrimal, tDual);

            Plato::ScalarVector tTotalDerivative("Total Derivative", aControl.size());
            this->calculateGradientControl(aName, aControl, tPrimal, tDual, tTotalDerivative);

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
            Plato::Dual tDual;
            tDual.scalar("step", 1.0);
            this->setDualVariables(tDual);

            Plato::Primal tPrimal;
            tPrimal.scalar("step", 1.0);
            this->setPrimalStates(tPrimal);

            this->calculateElemCharacteristicSize(tPrimal);
            this->calculateCriticalTimeStep(tPrimal);

            this->calculateCorrectorAdjoint(aName, aControl, tPrimal, tDual);
            this->calculateTemperatureAdjoint(aName, aControl, tPrimal, tDual);
            this->calculatePressureAdjoint(aName, aControl, tPrimal, tDual);
            this->calculatePredictorAdjoint(aControl, tPrimal, tDual);

            Plato::ScalarVector tTotalDerivative("Total Derivative", aControl.size());
            this->calculateGradientConfig(aName, aControl, tPrimal, tDual, tTotalDerivative);

            return tTotalDerivative;
        }
        else
        {
            THROWERR(std::string("Criterion with tag '") + aName + "' is not supported");
        }
    }

private:
    Plato::Solutions setOutputSolution()
    {
	Plato::Solutions tSolution;
        tSolution.set("velocity", mVelocity);
        tSolution.set("pressure", mPressure);
        if(mCalculateHeatTransfer)
        {
            tSolution.set("temperature", mTemperature);
        }
	return tSolution;
    }

    void setInitialConditions
    (Plato::Primal & aVariables)
    {
        const auto tTime = 0.0;
        const auto tPrevStep = 0;

        Plato::ScalarVector tVelBcValues;
        Plato::LocalOrdinalVector tVelBcDofs;
        mVelocityEssentialBCs.get(tVelBcDofs, tVelBcValues, tTime);
        auto tPreviouVel = Kokkos::subview(mVelocity, tPrevStep, Kokkos::ALL());
        Plato::cbs::enforce_boundary_condition(tVelBcDofs, tVelBcValues, tPreviouVel);
        aVariables.vector("previous velocity", tPreviouVel);

        Plato::ScalarVector tPressBcValues;
        Plato::LocalOrdinalVector tPressBcDofs;
        mPressureEssentialBCs.get(tPressBcDofs, tPressBcValues, tTime);
        auto tPreviousPress = Kokkos::subview(mPressure, tPrevStep, Kokkos::ALL());
        Plato::cbs::enforce_boundary_condition(tPressBcDofs, tPressBcValues, tPreviousPress);
        aVariables.vector("previous pressure", tPreviousPress);

        if(mCalculateHeatTransfer)
        {
            Plato::ScalarVector tTempBcValues;
            Plato::LocalOrdinalVector tTempBcDofs;
            mTemperatureEssentialBCs.get(tTempBcDofs, tTempBcValues, tTime);
            auto tPreviousTemp  = Kokkos::subview(mTemperature, tPrevStep, Kokkos::ALL());
            Plato::cbs::enforce_boundary_condition(tTempBcDofs, tTempBcValues, tPreviousTemp);
            aVariables.vector("previous temperature", tPreviousTemp);
        }
    }

    void printIteration
    (const Plato::Primal & aVariables)
    {
        if(Plato::Comm::rank(mMachine) == 0)
        {
            if(mPrintDiagnostics)
            {
                std::stringstream tMsg;
                auto tCriticalTimeStep = aVariables.vector("critical time step");
                auto tHostCriticalTimeStep = Kokkos::create_mirror(tCriticalTimeStep);
                Kokkos::deep_copy(tHostCriticalTimeStep, tCriticalTimeStep);
                const Plato::OrdinalType tIteration = aVariables.scalar("iteration");
                tMsg << "*************************************************************************************\n";
                tMsg << "* Critical Time Step: " << tHostCriticalTimeStep(0) << "\n";
                tMsg << "* CFD Quasi-Implicit Solver Iteration: " << tIteration << "\n";
                tMsg << "*************************************************************************************\n";
                Plato::append_text_to_file(tMsg, mDiagnostics);
            }
        }
    }

    void areDianosticsEnabled
    (Teuchos::ParameterList & aInputs)
    {
        mPrintDiagnostics = aInputs.get<bool>("Diagnostics", true);
        auto tFileName = aInputs.get<std::string>("Diagnostics File Name", "cfd_solver_diagnostics.txt");
        if(Plato::Comm::rank(mMachine) == 0)
        {
            Plato::open_text_file(tFileName, mDiagnostics, mPrintDiagnostics);
        }
    }

    void initialize
    (Teuchos::ParameterList & aInputs)
    {
        this->allocateCriteriaList(aInputs);
        this->allocateMemberStates(aInputs);
        this->areDianosticsEnabled(aInputs);
        this->parseNewtonSolverInputs(aInputs);
        this->parseConvergenceCriteria(aInputs);
        this->parseTimeIntegratorInputs(aInputs);
        this->parseHeatTransferEquation(aInputs);

    }

    void parseHeatTransferEquation
    (Teuchos::ParameterList & aInputs)
    {
        mCalculateHeatTransfer = Plato::Fluids::calculate_heat_transfer(aInputs);

        if(mCalculateHeatTransfer)
        {
            mTemperatureResidual =
                std::make_shared<Plato::Fluids::VectorFunction<typename PhysicsT::EnergyPhysicsT>>("Temperature", mSpatialModel, mDataMap, aInputs);
            mPrandtlNumber = Plato::Fluids::dimensionless_prandtl_number(aInputs);
        }
    }

    void parseNewtonSolverInputs
    (Teuchos::ParameterList & aInputs)
    {
        if(aInputs.isSublist("Newton Iteration"))
        {
            auto tNewtonIteration = aInputs.sublist("Newton Iteration");
            mPressureTolerance = tNewtonIteration.get<Plato::Scalar>("Pressure Tolerance", 1e-4);
            mPredictorTolerance = tNewtonIteration.get<Plato::Scalar>("Predictor Tolerance", 1e-4);
            mCorrectorTolerance = tNewtonIteration.get<Plato::Scalar>("Corrector Tolerance", 1e-4);
            mTemperatureTolerance = tNewtonIteration.get<Plato::Scalar>("Temperature Tolerance", 1e-4);
            mMaxPressureIterations = tNewtonIteration.get<Plato::OrdinalType>("Pressure Iterations", 10);
            mMaxPredictorIterations = tNewtonIteration.get<Plato::OrdinalType>("Predictor Iterations", 10);
            mMaxCorrectorIterations = tNewtonIteration.get<Plato::OrdinalType>("Corrector Iterations", 10);
            mMaxTemperatureIterations = tNewtonIteration.get<Plato::OrdinalType>("Temperature Iterations", 10);
        }
    }

    void parseTimeIntegratorInputs
    (Teuchos::ParameterList & aInputs)
    {
        if(aInputs.isSublist("Time Integration"))
        {
            auto tTimeIntegration = aInputs.sublist("Time Integration");
            mTimeStepSafetyFactor = tTimeIntegration.get<Plato::Scalar>("Safety Factor", 0.9);
        }
    }

    void parseConvergenceCriteria
    (Teuchos::ParameterList & aInputs)
    {
        if(aInputs.isSublist("Convergence"))
        {
            auto tConvergence = aInputs.sublist("Convergence");
            mSteadyStateTolerance = tConvergence.get<Plato::Scalar>("Steady State Tolerance", 1e-5);
            mMaxSteadyStateIterations = tConvergence.get<Plato::OrdinalType>("Maximum Iterations", 2000);
        }
    }

    void checkProblemSetup()
    {
        if(mPressureEssentialBCs.empty())
        {
            THROWERR("Pressure essential boundary conditions are not defined.")
        }
        if(mVelocityEssentialBCs.empty())
        {
            THROWERR("Velocity essential boundary conditions are not defined.")
        }
        if(mCalculateHeatTransfer)
        {
            if(mTemperatureEssentialBCs.empty())
            {
                THROWERR("Temperature essential boundary conditions are not defined.")
            }
            if(mTemperatureResidual.use_count() == 0)
            {
                THROWERR("Heat transfer calculation requested but temperature 'Vector Function' is not allocated.")
            }
        }
    }

    void allocateMemberStates(Teuchos::ParameterList & aInputs)
    {
        constexpr auto tTimeSnapshotsStored = 2;
        auto tNumNodes = mSpatialModel.Mesh.nverts();
        mPressure    = Plato::ScalarMultiVector("Pressure Snapshots", tTimeSnapshotsStored, tNumNodes);
        mTemperature = Plato::ScalarMultiVector("Temperature Snapshots", tTimeSnapshotsStored, tNumNodes);
        mVelocity    = Plato::ScalarMultiVector("Velocity Snapshots", tTimeSnapshotsStored, tNumNodes * mNumVelDofsPerNode);
        mPredictor   = Plato::ScalarMultiVector("Predictor Snapshots", tTimeSnapshotsStored, tNumNodes * mNumVelDofsPerNode);
    }

    void allocateCriteriaList(Teuchos::ParameterList &aInputs)
    {
        if(aInputs.isSublist("Criteria"))
        {
            Plato::Fluids::CriterionFactory<PhysicsT> tScalarFuncFactory;

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

    Plato::Scalar calculateVelocityMisfitNorm(const Plato::Primal & aVariables)
    {
        auto tNumNodes = mSpatialModel.Mesh.nverts();
        auto tCurrentVelocity = aVariables.vector("current velocity");
        auto tPreviousVelocity = aVariables.vector("previous velocity");
        auto tMisfitError = Plato::cbs::calculate_misfit_euclidean_norm<mNumVelDofsPerNode>(tNumNodes, tCurrentVelocity, tPreviousVelocity);
        auto tCurrentVelNorm = Plato::blas1::norm(tCurrentVelocity);
        auto tOutput = tMisfitError / tCurrentVelNorm;
        return tOutput;
    }

    Plato::Scalar calculatePressureMisfitNorm(const Plato::Primal & aVariables)
    {
        auto tNumNodes = mSpatialModel.Mesh.nverts();
        auto tCurrentPressure = aVariables.vector("current pressure");
        auto tPreviousPressure = aVariables.vector("previous pressure");
        auto tMisfitError = Plato::cbs::calculate_misfit_euclidean_norm<mNumPressDofsPerNode>(tNumNodes, tCurrentPressure, tPreviousPressure);
        auto tCurrentNorm = Plato::blas1::norm(tCurrentPressure);
        auto tOutput = tMisfitError / tCurrentNorm;
        return tOutput;
    }

    void printSteadyStateCriterion
    (const Plato::Primal & aVariables)
    {
        if(Plato::Comm::rank(mMachine) == 0)
        {
            if(mPrintDiagnostics)
            {
                std::stringstream tMsg;
                auto tCriterion = aVariables.scalar("current steady state criterion");
                tMsg << "\n-------------------------------------------------------------------------------------\n";
                tMsg << std::scientific << " Steady State Convergence: " << tCriterion << "\n";
                tMsg << "-------------------------------------------------------------------------------------\n\n";
                Plato::append_text_to_file(tMsg, mDiagnostics);
            }
        }
    }

    bool isFluidSolverDiverging
    (Plato::Primal & aVariables)
    {
        const Plato::OrdinalType tIteration = aVariables.scalar("iteration");
        if(tIteration <= 1)
        {
            aVariables.scalar("divergence count", 0);
            return false;
        }

        auto tCurrentCriterion = aVariables.scalar("current steady state criterion");
        if(!std::isfinite(tCurrentCriterion) || std::isnan(tCurrentCriterion))
        {
            return true;
        }
        return false;
    }

    bool checkStoppingCriteria
    (Plato::Primal & aVariables)
    {
        bool tStop = false;
        const Plato::OrdinalType tIteration = aVariables.scalar("iteration");
        const auto tCriterionValue = this->calculatePressureMisfitNorm(aVariables);
        aVariables.scalar("current steady state criterion", tCriterionValue);
        this->printSteadyStateCriterion(aVariables);


        if (tCriterionValue < mSteadyStateTolerance)
        {
            tStop = true;
        }
        else if (tIteration >= mMaxSteadyStateIterations)
        {
            tStop = true;
        }
        else if(this->isFluidSolverDiverging(aVariables))
        {
            tStop = true;
        }

        aVariables.scalar("previous steady state criterion", tCriterionValue);

        return tStop;
    }

    void calculateElemCharacteristicSize(Plato::Primal & aVariables)
    {
        auto tElemCharSizes =
            Plato::cbs::calculate_element_characteristic_sizes<mNumSpatialDims,mNumNodesPerCell>(mSpatialModel);
        aVariables.vector("element characteristic size", tElemCharSizes);
    }

    Plato::ScalarVector criticalTimeStep
    (const Plato::Primal & aVariables,
     const Plato::ScalarVector & aPreviousVelocity)
    {
        auto tElemCharSize = aVariables.vector("element characteristic size");
        auto tConvectiveVel = 
	    Plato::cbs::calculate_convective_velocity_magnitude<mNumNodesPerCell>(mSpatialModel, aPreviousVelocity);
        auto tCriticalConvectiveTimeStep = Plato::cbs::calculate_critical_convective_time_step
            (mSpatialModel, tElemCharSize, tConvectiveVel, mTimeStepSafetyFactor);

        Plato::ScalarVector tCriticalTimeStep("critical time step", 1);
        auto tHostCriticalTimeStep = Kokkos::create_mirror(tCriticalTimeStep);
        if(mCalculateHeatTransfer)
        {
	    auto tCriticalThermalTimeStep = Plato::cbs::calculate_critical_thermal_time_step
	        (tElemCharSize, mPrandtlNumber, mTimeStepSafetyFactor);
            auto tMinCriticalTimeStep = std::min(tCriticalConvectiveTimeStep, tCriticalThermalTimeStep);
            tHostCriticalTimeStep(0) = tMinCriticalTimeStep;
            Kokkos::deep_copy(tCriticalTimeStep, tHostCriticalTimeStep);
        }
        else
        {
            tHostCriticalTimeStep(0) = tCriticalConvectiveTimeStep;
            Kokkos::deep_copy(tCriticalTimeStep, tHostCriticalTimeStep);
        }
        return tCriticalTimeStep;
    }

    Plato::ScalarVector initialCriticalTimeStep
    (const Plato::Primal & aVariables)
    {
        Plato::ScalarVector tBcValues;
        Plato::LocalOrdinalVector tBcDofs;
        mVelocityEssentialBCs.get(tBcDofs, tBcValues);
        auto tPreviousVelocity = aVariables.vector("previous velocity");
        Plato::ScalarVector tInitialVelocity("initial velocity", tPreviousVelocity.size());
        Plato::blas1::update(1.0, tPreviousVelocity, 0.0, tInitialVelocity);
        Plato::cbs::enforce_boundary_condition(tBcDofs, tBcValues, tInitialVelocity);
        auto tCriticalTimeStep = this->criticalTimeStep(aVariables, tInitialVelocity);
        return tCriticalTimeStep;
    }

    void calculateCriticalTimeStep(Plato::Primal & aVariables)
    {
        auto tIteration = aVariables.scalar("iteration");
        if(tIteration > 1)
        {
            auto tPreviousVelocity = aVariables.vector("previous velocity");
            auto tCriticalTimeStep = this->criticalTimeStep(aVariables, tPreviousVelocity);
            aVariables.vector("critical time step", tCriticalTimeStep);
        }
        else
        {
            auto tCriticalTimeStep = this->initialCriticalTimeStep(aVariables);
            aVariables.vector("critical time step", tCriticalTimeStep);
        }
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

    void updatePreviousStates(Plato::Primal & aVariables)
    {
        constexpr Plato::OrdinalType tPrevState = 0;

        auto tCurrentVelocity = aVariables.vector("current velocity");
        auto tPreviousVelocity = Kokkos::subview(mVelocity, tPrevState, Kokkos::ALL());
        Plato::blas1::copy(tCurrentVelocity, tPreviousVelocity);

        auto tCurrentPressure = aVariables.vector("current pressure");
        auto tPreviousPressure = Kokkos::subview(mPressure, tPrevState, Kokkos::ALL());
        Plato::blas1::copy(tCurrentPressure, tPreviousPressure);

        auto tCurrentPredictor = aVariables.vector("current predictor");
        auto tPreviousPredictor = Kokkos::subview(mPredictor, tPrevState, Kokkos::ALL());
        Plato::blas1::copy(tCurrentPredictor, tPreviousPredictor);

        if(mCalculateHeatTransfer)
        {
            auto tCurrentTemperature = aVariables.vector("current temperature");
            auto tPreviousTemperature = Kokkos::subview(mTemperature, tPrevState, Kokkos::ALL());
            Plato::blas1::copy(tCurrentTemperature, tPreviousTemperature);
        }
    }

    void setPrimalStates(Plato::Primal & aVariables)
    {
        constexpr Plato::OrdinalType tCurrentState = 1;
        auto tCurrentVel   = Kokkos::subview(mVelocity, tCurrentState, Kokkos::ALL());
        auto tCurrentPred  = Kokkos::subview(mPredictor, tCurrentState, Kokkos::ALL());
        auto tCurrentPress = Kokkos::subview(mPressure, tCurrentState, Kokkos::ALL());
        aVariables.vector("current velocity", tCurrentVel);
        aVariables.vector("current pressure", tCurrentPress);
        aVariables.vector("current predictor", tCurrentPred);

        constexpr auto tPrevState = tCurrentState - 1;
        auto tPreviouVel = Kokkos::subview(mVelocity, tPrevState, Kokkos::ALL());
        auto tPreviousPred = Kokkos::subview(mPredictor, tPrevState, Kokkos::ALL());
        auto tPreviousPress = Kokkos::subview(mPressure, tPrevState, Kokkos::ALL());
        aVariables.vector("previous velocity", tPreviouVel);
        aVariables.vector("previous predictor", tPreviousPred);
        aVariables.vector("previous pressure", tPreviousPress);

        auto tCurrentTemp = Kokkos::subview(mTemperature, tCurrentState, Kokkos::ALL());
        aVariables.vector("current temperature", tCurrentTemp);
        auto tPreviousTemp = Kokkos::subview(mTemperature, tPrevState, Kokkos::ALL());
        aVariables.vector("previous temperature", tPreviousTemp);
    }

    void printCorrectorSolverHeader()
    {
        if(Plato::Comm::rank(mMachine) == 0)
        {
            if(mPrintDiagnostics)
            {
                std::stringstream tMsg;
                tMsg << "\n-------------------------------------------------------------------------------------\n";
                tMsg << "*                           Momentum Corrector Solver                               *\n";
                tMsg << "-------------------------------------------------------------------------------------\n";
                Plato::append_text_to_file(tMsg, mDiagnostics);
            }
        }
    }

    void updateCorrector
    (const Plato::ScalarVector & aControl,
           Plato::Primal       & aStates)
    {
        this->printCorrectorSolverHeader();
        this->printNewtonHeader();

        auto tCurrentVelocity = aStates.vector("current velocity");
        Plato::blas1::fill(0.0, tCurrentVelocity);

        // calculate current residual and jacobian matrix
        auto tResidual = mCorrectorResidual.value(aControl, aStates);
        auto tJacobian = mCorrectorResidual.gradientCurrentVel(aControl, aStates);

        // apply constraints
        Plato::ScalarVector tBcValues;
        Plato::LocalOrdinalVector tBcDofs;
        mVelocityEssentialBCs.get(tBcDofs, tBcValues);

        // create linear solver
        auto tParamList = mInputs.sublist("Linear Solver");
        Plato::SolverFactory tSolverFactory(tParamList);
        auto tSolver = tSolverFactory.create(mSpatialModel.Mesh, mMachine, mNumVelDofsPerNode);

        // set initial guess for current velocity
        Plato::OrdinalType tIteration = 1;
        Plato::Scalar tInitialNormStep = 0.0;
        Plato::ScalarVector tDeltaCorrector("delta corrector", tCurrentVelocity.size());
        while(true)
        {
            aStates.scalar("newton iteration", tIteration);

            Plato::blas1::fill(0.0, tDeltaCorrector);
            Plato::blas1::scale(-1.0, tResidual);
            tSolver->solve(*tJacobian, tDeltaCorrector, tResidual);
            Plato::blas1::update(1.0, tDeltaCorrector, 1.0, tCurrentVelocity);

            auto tNormResidual = Plato::blas1::norm(tResidual);
            aStates.scalar("norm residual", tNormResidual);
            auto tNormStep = Plato::blas1::norm(tDeltaCorrector);
            if(tIteration <= 1)
            {
                tInitialNormStep = tNormStep;
            }
            tNormStep = tNormStep / tInitialNormStep;
            aStates.scalar("norm step", tNormStep);

            this->printNewtonDiagnostics(aStates);
            if(tNormStep <= mCorrectorTolerance || tIteration >= mMaxCorrectorIterations)
            {
                break;
            }

            // calculate current residual and jacobian matrix
            tResidual = mCorrectorResidual.value(aControl, aStates);

            tIteration++;
        }
        Plato::cbs::enforce_boundary_condition(tBcDofs, tBcValues, tCurrentVelocity);
    }

    void printNewtonHeader()
    {
        if(Plato::Comm::rank(mMachine) == 0)
        {
            if(mPrintDiagnostics)
            {
                std::stringstream tMsg;
                tMsg << "Iteration" << std::setw(16) << "Delta(u*)" << std::setw(18) << "Residual\n";
                Plato::append_text_to_file(tMsg, mDiagnostics);
            }
        }
    }

    void printPredictorSolverHeader()
    {
        if(Plato::Comm::rank(mMachine) == 0)
        {
            if(mPrintDiagnostics)
            {
                std::stringstream tMsg;
                tMsg << "\n-------------------------------------------------------------------------------------\n";
                tMsg << "*                           Momentum Predictor Solver                               *\n";
                tMsg << "-------------------------------------------------------------------------------------\n";
                Plato::append_text_to_file(tMsg, mDiagnostics);
            }
        }
    }

    void printNewtonDiagnostics
    (Plato::Primal & aVariables)
    {
        if(Plato::Comm::rank(mMachine) == 0)
        {
            if(mPrintDiagnostics)
            {
                std::stringstream tMsg;
                auto tNormStep = aVariables.scalar("norm step");
                auto tNormResidual = aVariables.scalar("norm residual");
                Plato::OrdinalType tIteration = aVariables.scalar("newton iteration");
                tMsg << tIteration << std::setw(24) << std::scientific << tNormStep << std::setw(18) << tNormResidual << "\n";
                Plato::append_text_to_file(tMsg, mDiagnostics);
            }
        }
    }

    void updatePredictor
    (const Plato::ScalarVector & aControl,
           Plato::Primal       & aStates)
    {
        this->printPredictorSolverHeader();
        this->printNewtonHeader();

        auto tCurrentPredictor = aStates.vector("current predictor");
        Plato::blas1::fill(0.0, tCurrentPredictor);

        // calculate current residual and jacobian matrix
        auto tResidual = mPredictorResidual.value(aControl, aStates);
        auto tJacobian = mPredictorResidual.gradientPredictor(aControl, aStates);

        // prepare constraints dofs
        Plato::ScalarVector tBcValues;
        Plato::LocalOrdinalVector tBcDofs;
        mVelocityEssentialBCs.get(tBcDofs, tBcValues);

        // create linear solver
        auto tParamList = mInputs.sublist("Linear Solver");
        Plato::SolverFactory tSolverFactory(tParamList);
        auto tSolver = tSolverFactory.create(mSpatialModel.Mesh, mMachine, mNumVelDofsPerNode);

        Plato::OrdinalType tIteration = 1;
        Plato::Scalar tInitialNormStep = 0.0;
        Plato::ScalarVector tDeltaPredictor("delta predictor", tCurrentPredictor.size());
        while(true)
        {
            aStates.scalar("newton iteration", tIteration);

            Plato::blas1::fill(0.0, tDeltaPredictor);
            Plato::blas1::scale(-1.0, tResidual);
            tSolver->solve(*tJacobian, tDeltaPredictor, tResidual);
            Plato::blas1::update(1.0, tDeltaPredictor, 1.0, tCurrentPredictor);

            auto tNormResidual = Plato::blas1::norm(tResidual);
            aStates.scalar("norm residual", tNormResidual);
            auto tNormStep = Plato::blas1::norm(tDeltaPredictor);
            if(tIteration <= 1)
            {
                tInitialNormStep = tNormStep;
            }
            tNormStep = tNormStep / tInitialNormStep;
            aStates.scalar("norm step", tNormStep);

            this->printNewtonDiagnostics(aStates);
            if(tNormStep <= mPredictorTolerance || tIteration >= mMaxPredictorIterations)
            {
                break;
            }

            tResidual = mPredictorResidual.value(aControl, aStates);

            tIteration++;
        }
    }

    void printPressureSolverHeader()
    {
        if(Plato::Comm::rank(mMachine) == 0)
        {
            if(mPrintDiagnostics)
            {
                std::stringstream tMsg;
                tMsg << "\n-------------------------------------------------------------------------------------\n";
                tMsg << "*                                Pressure Solver                                    *\n";
                tMsg << "-------------------------------------------------------------------------------------\n";
                Plato::append_text_to_file(tMsg, mDiagnostics);
            }
        }
    }

    void updatePressure
    (const Plato::ScalarVector & aControl,
           Plato::Primal       & aStates)
    {
        this->printPressureSolverHeader();
        this->printNewtonHeader();

        auto tCurrentPressure = aStates.vector("current pressure");
        Plato::blas1::fill(0.0, tCurrentPressure);

        // calculate current residual and jacobian matrix
        auto tResidual = mPressureResidual.value(aControl, aStates);
        auto tJacobian = mPressureResidual.gradientCurrentPress(aControl, aStates);

        // prepare constraints dofs
        Plato::ScalarVector tBcValues;
        Plato::LocalOrdinalVector tBcDofs;
        mPressureEssentialBCs.get(tBcDofs, tBcValues);

        // create linear solver
        auto tParamList = mInputs.sublist("Linear Solver");
        Plato::SolverFactory tSolverFactory(tParamList);
        auto tSolver = tSolverFactory.create(mSpatialModel.Mesh, mMachine, mNumPressDofsPerNode);

        Plato::OrdinalType tIteration = 1;
        Plato::Scalar tInitialNormStep = 0.0;
        Plato::ScalarVector tDeltaPressure("delta pressure", tCurrentPressure.size());
        while(true)
        {
            aStates.scalar("newton iteration", tIteration);

            Plato::blas1::fill(0.0, tDeltaPressure);
            Plato::blas1::scale(-1.0, tResidual);
            Plato::apply_constraints<mNumPressDofsPerNode>(tBcDofs, tBcValues, tJacobian, tResidual);
            tSolver->solve(*tJacobian, tDeltaPressure, tResidual);
            Plato::blas1::update(1.0, tDeltaPressure, 1.0, tCurrentPressure);

            auto tNormResidual = Plato::blas1::norm(tResidual);
            aStates.scalar("norm residual", tNormResidual);
            auto tNormStep = Plato::blas1::norm(tDeltaPressure);
            if(tIteration <= 1)
            {
                tInitialNormStep = tNormStep;
            }
            tNormStep = tNormStep / tInitialNormStep;
            aStates.scalar("norm step", tNormStep);

            this->printNewtonDiagnostics(aStates);
            if(tNormStep <= mPressureTolerance || tIteration >= mMaxPressureIterations)
            {
                break;
            }

            // calculate current residual and jacobian matrix
            tJacobian = mPressureResidual.gradientCurrentPress(aControl, aStates);
            tResidual = mPressureResidual.value(aControl, aStates);

            tIteration++;
        }
    }

    void printTemperatureSolverHeader()
    {
        if(Plato::Comm::rank(mMachine) == 0)
        {
            if(mPrintDiagnostics)
            {
                std::stringstream tMsg;
                tMsg << "\n-------------------------------------------------------------------------------------\n";
                tMsg << "*                             Temperature Solver                                    *\n";
                tMsg << "-------------------------------------------------------------------------------------\n";
                Plato::append_text_to_file(tMsg, mDiagnostics);
            }
        }
    }

    void updateTemperature
    (const Plato::ScalarVector & aControl,
           Plato::Primal       & aStates)
    {
        this->printTemperatureSolverHeader();
        this->printNewtonHeader();

        auto tCurrentTemperature = aStates.vector("current temperature");
        Plato::blas1::fill(0.0, tCurrentTemperature);

        // calculate current residual and jacobian matrix
        auto tResidual = mTemperatureResidual->value(aControl, aStates);
        auto tJacobian = mTemperatureResidual->gradientCurrentTemp(aControl, aStates);

        // apply constraints
        Plato::ScalarVector tBcValues;
        Plato::LocalOrdinalVector tBcDofs;
        mTemperatureEssentialBCs.get(tBcDofs, tBcValues);

        // solve energy equation (consistent or mass lumped)
        auto tParamList = mInputs.sublist("Linear Solver");
        Plato::SolverFactory tSolverFactory(tParamList);
        auto tSolver = tSolverFactory.create(mSpatialModel.Mesh, mMachine, mNumTempDofsPerNode);

        Plato::OrdinalType tIteration = 1;
        Plato::Scalar tInitialNormStep = 0.0;
        Plato::ScalarVector tDeltaTemperature("delta temperature", tCurrentTemperature.size());
        while(true)
        {
            aStates.scalar("newton iteration", tIteration);

            Plato::blas1::fill(0.0, tDeltaTemperature);
            Plato::blas1::scale(-1.0, tResidual);
            tSolver->solve(*tJacobian, tDeltaTemperature, tResidual);
            Plato::blas1::update(1.0, tDeltaTemperature, 1.0, tCurrentTemperature);

            auto tNormResidual = Plato::blas1::norm(tResidual);
            aStates.scalar("norm residual", tNormResidual);
            auto tNormStep = Plato::blas1::norm(tDeltaTemperature);
            if(tIteration <= 1)
            {
                tInitialNormStep = tNormStep;
            }
            tNormStep = tNormStep / tInitialNormStep;
            aStates.scalar("norm step", tNormStep);

            this->printNewtonDiagnostics(aStates);
            if(tNormResidual < mTemperatureTolerance || tIteration >= mMaxTemperatureIterations)
            {
                break;
            }

            // calculate current residual and jacobian matrix
            tResidual = mTemperatureResidual->value(aControl, aStates);

            tIteration++;
        }
        Plato::cbs::enforce_boundary_condition(tBcDofs, tBcValues, tCurrentTemperature);
    }

    void calculatePredictorAdjoint
    (const Plato::ScalarVector & aControl,
     const Plato::Primal       & aPrimal,
           Plato::Dual         & aDual)
    {
        auto tCurrentPredictorAdjoint = aDual.vector("current predictor adjoint");
        Plato::blas1::fill(0.0, tCurrentPredictorAdjoint);

        auto tCurrentVelocityAdjoint = aDual.vector("current velocity adjoint");
        auto tGradResVelWrtPredictor = mCorrectorResidual.gradientPredictor(aControl, aPrimal);
        Plato::ScalarVector tRHS("right hand side", tCurrentVelocityAdjoint.size());
        Plato::MatrixTimesVectorPlusVector(tGradResVelWrtPredictor, tCurrentVelocityAdjoint, tRHS);

        auto tCurrentPressureAdjoint = aDual.vector("current pressure adjoint");
        auto tGradResPressWrtPredictor = mPressureResidual.gradientPredictor(aControl, aPrimal);
        Plato::MatrixTimesVectorPlusVector(tGradResPressWrtPredictor, tCurrentPressureAdjoint, tRHS);
        Plato::blas1::scale(-1.0, tRHS);

        auto tParamList = mInputs.sublist("Linear Solver");
        Plato::SolverFactory tSolverFactory(tParamList);
        auto tSolver = tSolverFactory.create(mSpatialModel.Mesh, mMachine, mNumVelDofsPerNode);
        auto tJacobianPredictor = mPredictorResidual.gradientPredictor(aControl, aPrimal);
        tSolver->solve(*tJacobianPredictor, tCurrentPredictorAdjoint, tRHS);
    }

    void calculatePressureAdjoint
    (const std::string         & aName,
     const Plato::ScalarVector & aControl,
     const Plato::Primal       & aPrimal,
           Plato::Dual         & aDual)
    {
        auto tCurrentPressAdjoint = aDual.vector("current pressure adjoint");
        Plato::blas1::fill(0.0, tCurrentPressAdjoint);

        auto tRHS = mCriteria[aName]->gradientCurrentPress(aControl, aPrimal);
        auto tGradResVelWrtCurPress = mCorrectorResidual.gradientCurrentPress(aControl, aPrimal);
        auto tCurrentVelocityAdjoint = aDual.vector("current velocity adjoint");
        Plato::MatrixTimesVectorPlusVector(tGradResVelWrtCurPress, tCurrentVelocityAdjoint, tRHS);
        Plato::blas1::scale(-1.0, tRHS);

        auto tParamList = mInputs.sublist("Linear Solver");
        Plato::SolverFactory tSolverFactory(tParamList);
        auto tSolver = tSolverFactory.create(mSpatialModel.Mesh, mMachine, mNumPressDofsPerNode);
        auto tJacobianPressure = mPressureResidual.gradientCurrentPress(aControl, aPrimal);
        tSolver->solve(*tJacobianPressure, tCurrentPressAdjoint, tRHS);
    }

    void calculateTemperatureAdjoint
    (const std::string         & aName,
     const Plato::ScalarVector & aControl,
     const Plato::Primal       & aPrimal,
           Plato::Dual         & aDual)
    {
        auto tCurrentTempAdjoint = aDual.vector("current temperature adjoint");
        Plato::blas1::fill(0.0, tCurrentTempAdjoint);

        auto tRHS = mCriteria[aName]->gradientCurrentTemp(aControl, aPrimal);
        Plato::blas1::scale(-1.0, tRHS);

        auto tParamList = mInputs.sublist("Linear Solver");
        Plato::SolverFactory tSolverFactory(tParamList);
        auto tSolver = tSolverFactory.create(mSpatialModel.Mesh, mMachine, mNumTempDofsPerNode);
        auto tJacobianTemperature = mTemperatureResidual->gradientCurrentTemp(aControl, aPrimal);
        tSolver->solve(*tJacobianTemperature, tCurrentTempAdjoint, tRHS);
    }

    void calculateCorrectorAdjoint
    (const std::string         & aName,
     const Plato::ScalarVector & aControl,
     const Plato::Primal       & aPrimal,
           Plato::Dual         & aDual)
    {
        auto tCurrentVelocityAdjoint = aDual.vector("current velocity adjoint");
        Plato::blas1::fill(0.0, tCurrentVelocityAdjoint);

        auto tRHS = mCriteria[aName]->gradientCurrentVel(aControl, aPrimal);
        Plato::blas1::scale(-1.0, tRHS);

        auto tParamList = mInputs.sublist("Linear Solver");
        Plato::SolverFactory tSolverFactory(tParamList);
        auto tSolver = tSolverFactory.create(mSpatialModel.Mesh, mMachine, mNumVelDofsPerNode);
        auto tJacobianVelocity = mCorrectorResidual.gradientCurrentVel(aControl, aPrimal);
        tSolver->solve(*tJacobianVelocity, tCurrentVelocityAdjoint, tRHS);
    }

    void calculateGradientControl
    (const std::string         & aName,
     const Plato::ScalarVector & aControl,
     const Plato::Primal       & aPrimal,
     const Plato::Dual         & aDual,
           Plato::ScalarVector & aTotalDerivative)
    {
        auto tGradCriterionWrtControl = mCriteria[aName]->gradientControl(aControl, aPrimal);

        auto tCurrentPredictorAdjoint = aDual.vector("current predictor adjoint");
        auto tGradResPredWrtControl = mPredictorResidual.gradientControl(aControl, aPrimal);
        Plato::MatrixTimesVectorPlusVector(tGradResPredWrtControl, tCurrentPredictorAdjoint, tGradCriterionWrtControl);

        auto tCurrentPressureAdjoint = aDual.vector("current pressure adjoint");
        auto tGradResPressWrtControl = mPressureResidual.gradientControl(aControl, aPrimal);
        Plato::MatrixTimesVectorPlusVector(tGradResPressWrtControl, tCurrentPressureAdjoint, tGradCriterionWrtControl);

        auto tCurrentTemperatureAdjoint = aDual.vector("current temperature adjoint");
        auto tGradResTempWrtControl = mTemperatureResidual->gradientControl(aControl, aPrimal);
        Plato::MatrixTimesVectorPlusVector(tGradResTempWrtControl, tCurrentTemperatureAdjoint, tGradCriterionWrtControl);

        auto tCurrentVelocityAdjoint = aDual.vector("current velocity adjoint");
        auto tGradResVelWrtControl = mCorrectorResidual.gradientControl(aControl, aPrimal);
        Plato::MatrixTimesVectorPlusVector(tGradResVelWrtControl, tCurrentVelocityAdjoint, tGradCriterionWrtControl);

        Plato::blas1::axpy(1.0, tGradCriterionWrtControl, aTotalDerivative);
    }

    void calculateGradientConfig
    (const std::string         & aName,
     const Plato::ScalarVector & aControl,
     const Plato::Primal       & aPrimal,
     const Plato::Dual         & aDual,
           Plato::ScalarVector & aTotalDerivative)
    {
        auto tGradCriterionWrtConfig = mCriteria[aName]->gradientConfig(aControl, aPrimal);

        auto tCurrentPredictorAdjoint = aDual.vector("current predictor adjoint");
        auto tGradResPredWrtConfig = mPredictorResidual.gradientConfig(aControl, aPrimal);
        Plato::MatrixTimesVectorPlusVector(tGradResPredWrtConfig, tCurrentPredictorAdjoint, tGradCriterionWrtConfig);

        auto tCurrentPressureAdjoint = aDual.vector("current pressure adjoint");
        auto tGradResPressWrtConfig = mPressureResidual.gradientConfig(aControl, aPrimal);
        Plato::MatrixTimesVectorPlusVector(tGradResPressWrtConfig, tCurrentPressureAdjoint, tGradCriterionWrtConfig);

        auto tCurrentTemperatureAdjoint = aDual.vector("current temperature adjoint");
        auto tGradResTempWrtConfig = mTemperatureResidual->gradientConfig(aControl, aPrimal);
        Plato::MatrixTimesVectorPlusVector(tGradResTempWrtConfig, tCurrentTemperatureAdjoint, tGradCriterionWrtConfig);

        auto tCurrentVelocityAdjoint = aDual.vector("current velocity adjoint");
        auto tGradResVelWrtConfig = mCorrectorResidual.gradientConfig(aControl, aPrimal);
        Plato::MatrixTimesVectorPlusVector(tGradResVelWrtConfig, tCurrentVelocityAdjoint, tGradCriterionWrtConfig);

        Plato::blas1::axpy(1.0, tGradCriterionWrtConfig, aTotalDerivative);
    }
};
// class QuasiImplicit

}
// namespace Hyperbolic

}
//namespace Plato

namespace ComputationalFluidDynamicsTests
{

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, IsothermalFlowOnChannel_Re100)
{
    // set xml file inputs
    Teuchos::RCP<Teuchos::ParameterList> tInputs =
        Teuchos::getParametersFromXmlString(
            "<ParameterList name='Plato Problem'>"
            "  <ParameterList name='Hyperbolic'>"
            "    <Parameter name='Heat Transfer' type='string' value='None'/>"
            "    <ParameterList  name='Dimensionless Properties'>"
            "      <Parameter  name='Reynolds Number'  type='double'  value='1e2'/>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList name='Spatial Model'>"
            "    <ParameterList name='Domains'>"
            "      <ParameterList name='Design Volume'>"
            "        <Parameter name='Element Block' type='string' value='body'/>"
            "        <Parameter name='Material Model' type='string' value='Water'/>"
            "      </ParameterList>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList  name='Velocity Essential Boundary Conditions'>"
            "    <ParameterList  name='X-Dir Inlet Velocity'>"
            "      <Parameter  name='Type'     type='string' value='Fixed Value'/>"
            "      <Parameter  name='Value'    type='double' value='1.0'/>"
            "      <Parameter  name='Index'    type='int'    value='0'/>"
            "      <Parameter  name='Sides'    type='string' value='x-'/>"
            "    </ParameterList>"
            "    <ParameterList  name='Y-Dir Inlet Velocity'>"
            "      <Parameter  name='Type'     type='string' value='Fixed Value'/>"
            "      <Parameter  name='Value'    type='double' value='0'/>"
            "      <Parameter  name='Index'    type='int'    value='1'/>"
            "      <Parameter  name='Sides'    type='string' value='x-'/>"
            "    </ParameterList>"
            "    <ParameterList  name='X-Dir No-Slip on Y+'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='0'/>"
            "      <Parameter  name='Sides'    type='string' value='y+'/>"
            "    </ParameterList>"
            "    <ParameterList  name='Y-Dir No-Slip on Y+'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='1'/>"
            "      <Parameter  name='Sides'    type='string' value='y+'/>"
            "    </ParameterList>"
            "    <ParameterList  name='X-Dir No-Slip on Y-'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='0'/>"
            "      <Parameter  name='Sides'    type='string' value='y-'/>"
            "    </ParameterList>"
            "    <ParameterList  name='Y-Dir No-Slip on Y-'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='1'/>"
            "      <Parameter  name='Sides'    type='string' value='y-'/>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList  name='Pressure Essential Boundary Conditions'>"
            "    <ParameterList  name='Outlet Pressure'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='0'/>"
            "      <Parameter  name='Sides'    type='string' value='x+'/>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList  name='Time Integration'>"
            "    <Parameter name='Safety Factor'      type='double' value='1.0'/>"
            "  </ParameterList>"
            "  <ParameterList  name='Linear Solver'>"
            "    <Parameter name='Solver Stack' type='string' value='Epetra'/>"
            "  </ParameterList>"
            "  <ParameterList  name='Convergence'>"
            "    <Parameter name='Maximum Iterations' type='int' value='1'/>"
            "    <Parameter name='Steady State Tolerance' type='double' value='1e-5'/>"
            "  </ParameterList>"
            "</ParameterList>"
            );

    // build mesh, spatial domain, and spatial model
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,2,2);
    auto tMeshSets = PlatoUtestHelpers::get_box_mesh_sets(tMesh.operator*());
    Plato::SpatialDomain tDomain(tMesh.operator*(), tMeshSets, "box");
    tDomain.cellOrdinals("body");

    // create communicator
    MPI_Comm tMyComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &tMyComm);
    Plato::Comm::Machine tMachine(tMyComm);

    // create and run incompressible cfd problem
    constexpr auto tSpaceDim = 2;
    Plato::Fluids::QuasiImplicit<Plato::IncompressibleFluids<tSpaceDim>> tProblem(*tMesh, tMeshSets, *tInputs, tMachine);
    const auto tNumVerts = tMesh->nverts();
    auto tControls = Plato::ScalarVector("Controls", tNumVerts);
    Plato::blas1::fill(1.0, tControls);
    auto tSolution = tProblem.solution(tControls);
    tProblem.output("cfd_test_problem");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, LidDrivenCavity_Re100)
{
    // set xml file inputs
    Teuchos::RCP<Teuchos::ParameterList> tInputs =
        Teuchos::getParametersFromXmlString(
            "<ParameterList name='Plato Problem'>"
            "  <ParameterList name='Hyperbolic'>"
            "    <Parameter name='Heat Transfer' type='string' value='None'/>"
            "    <ParameterList  name='Dimensionless Properties'>"
            "      <Parameter  name='Prandtl Number'   type='double'  value='1.7'/>"
            "      <Parameter  name='Reynolds Number'  type='double'  value='1e2'/>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList name='Spatial Model'>"
            "    <ParameterList name='Domains'>"
            "      <ParameterList name='Design Volume'>"
            "        <Parameter name='Element Block' type='string' value='body'/>"
            "        <Parameter name='Material Model' type='string' value='Water'/>"
            "      </ParameterList>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList name='Material Models'>"
            "    <ParameterList name='Water'>"
            "      <ParameterList name='Thermal Properties'>"
            "        <Parameter  name='Thermal Conductivity'  type='double'  value='1'/>"
            "      </ParameterList>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList  name='Velocity Essential Boundary Conditions'>"
            "    <ParameterList  name='X-Dir No-Slip on X-'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='0'/>"
            "      <Parameter  name='Sides'    type='string' value='x-'/>"
            "    </ParameterList>"
            "    <ParameterList  name='Y-Dir No-Slip on X-'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='1'/>"
            "      <Parameter  name='Sides'    type='string' value='x-'/>"
            "    </ParameterList>"
            "    <ParameterList  name='X-Dir No-Slip on X+'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='0'/>"
            "      <Parameter  name='Sides'    type='string' value='x+'/>"
            "    </ParameterList>"
            "    <ParameterList  name='Y-Dir No-Slip on X+'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='1'/>"
            "      <Parameter  name='Sides'    type='string' value='x+'/>"
            "    </ParameterList>"
            "    <ParameterList  name='X-Dir No-Slip on Y-'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='0'/>"
            "      <Parameter  name='Sides'    type='string' value='y-'/>"
            "    </ParameterList>"
            "    <ParameterList  name='Y-Dir No-Slip on Y-'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='1'/>"
            "      <Parameter  name='Sides'    type='string' value='y-'/>"
            "    </ParameterList>"
            "    <ParameterList  name='Tangential Velocity'>"
            "      <Parameter  name='Type'     type='string' value='Fixed Value'/>"
            "      <Parameter  name='Value'    type='double' value='1.0'/>"
            "      <Parameter  name='Index'    type='int'    value='0'/>"
            "      <Parameter  name='Sides'    type='string' value='y+'/>"
            "    </ParameterList>"
            "    <ParameterList  name='Normal Velocity'>"
            "      <Parameter  name='Type'     type='string' value='Fixed Value'/>"
            "      <Parameter  name='Value'    type='double' value='0'/>"
            "      <Parameter  name='Index'    type='int'    value='1'/>"
            "      <Parameter  name='Sides'    type='string' value='y-'/>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList  name='Pressure Essential Boundary Conditions'>"
            "    <ParameterList  name='Zero Pressure'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='0'/>"
            "      <Parameter  name='Sides'    type='string' value='pressure'/>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList  name='Newton Iteration'>"
            "    <Parameter name='Pressure Tolerance'  type='double' value='1e-4'/>"
            "    <Parameter name='Predictor Tolerance' type='double' value='1e-4'/>"
            "    <Parameter name='Corrector Tolerance' type='double' value='1e-4'/>"
            "  </ParameterList>"
            "  <ParameterList  name='Time Integration'>"
            "    <Parameter name='Safety Factor'      type='double' value='1.0'/>"
            "  </ParameterList>"
            "  <ParameterList  name='Linear Solver'>"
            "    <Parameter name='Solver Stack' type='string' value='Epetra'/>"
            "  </ParameterList>"
            "  <ParameterList  name='Convergence'>"
            "    <Parameter name='Steady State Tolerance' type='double' value='1e-5'/>"
            "  </ParameterList>"
            "</ParameterList>"
            );

    // build mesh, spatial domain, and spatial model
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,5,5);
    auto tMeshSets = PlatoUtestHelpers::get_box_mesh_sets(tMesh.operator*());
    Plato::SpatialDomain tDomain(tMesh.operator*(), tMeshSets, "box");
    tDomain.cellOrdinals("body");

    // add pressure essential boundary condition to node set list
    Omega_h::Write<int> tWritePress(1);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, 1), LAMBDA_EXPRESSION(const Plato::OrdinalType & aOrdinal)
    {
        tWritePress[aOrdinal]=0; 
    }, "set pressure bc dofs value");
    auto tPressBcNodeIds = Omega_h::LOs(tWritePress);
    tMeshSets[Omega_h::NODE_SET].insert( std::pair<std::string,Omega_h::LOs>("pressure",tPressBcNodeIds) );

    // create communicator
    MPI_Comm tMyComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &tMyComm);
    Plato::Comm::Machine tMachine(tMyComm);

    // create and run incompressible cfd problem
    constexpr auto tSpaceDim = 2;
    Plato::Fluids::QuasiImplicit<Plato::IncompressibleFluids<tSpaceDim>> tProblem(*tMesh, tMeshSets, *tInputs, tMachine);
    const auto tNumVerts = tMesh->nverts();
    auto tControls = Plato::ScalarVector("Controls", tNumVerts);
    Plato::blas1::fill(1.0, tControls);
    auto tSolution = tProblem.solution(tControls);
    //tProblem.output("cfd_test_problem");

    // test solution
    auto tTags = tSolution.tags();
    std::vector<std::string> tGoldTags = { "velocity", "pressure" };
    TEST_EQUALITY(tGoldTags.size(), tTags.size());
    for(auto& tTag : tTags)
    {
        auto tItr = std::find(tGoldTags.begin(), tGoldTags.end(), tTag);
        TEST_ASSERT(tItr != tGoldTags.end());
        TEST_EQUALITY(*tItr, tTag);
    }

    auto tTol = 1e-4;
    auto tPressure = tSolution.get("pressure");
    auto tPressSubView = Kokkos::subview(tPressure, 1, Kokkos::ALL());
    auto tHostPressure = Kokkos::create_mirror(tPressSubView);
    Kokkos::deep_copy(tHostPressure, tPressSubView);
    std::vector<double> tGoldPressure = 
    { 0.000000e+00, 3.731415e-03, 7.816292e-03, 7.587085e-03, 1.241365e-02, 1.063465e-02,
      -5.198417e-04, -3.482819e-03, -5.190182e-04, -6.584415e-02, -4.304338e-02, -2.550113e-01,
      -4.137012e-01, -2.028317e-01, -1.218144e-01, -6.249967e-02, -3.871247e-02, -2.076437e-02,
      6.898296e-03, 2.266167e-02, 3.563244e-02, 1.129117e-01, 1.519040e-01, 3.731144e-01,
      2.325776e-01, 4.197060e-02, 6.356792e-02, 2.115928e-02, 7.908880e-03, 3.050671e-03,
      4.394201e-03, 7.350702e-03, 1.146635e-02, 6.443506e-03, 1.890785e-02, 1.827038e-02 };
    for(auto& tGoldPress : tGoldPressure)
    {
        auto tDof = &tGoldPress - &tGoldPressure[0];
        TEST_FLOATING_EQUALITY(tGoldPress, tHostPressure(tDof), tTol);
    }    
    //Plato::print(tPressSubView, "steady state pressure");

    auto tVelocity = tSolution.get("velocity");
    auto tVelSubView = Kokkos::subview(tVelocity, 1, Kokkos::ALL());
    auto tHostVelocity = Kokkos::create_mirror(tVelSubView);
    Kokkos::deep_copy(tHostVelocity, tVelSubView);
    std::vector<double> tGoldVelocity = 
        { 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 2.833694e-02, 3.161706e-02,
          0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, -3.622390e-02, 7.679319e-03,
          -2.513879e-01, -2.315935e-02, -9.600695e-02, 8.900196e-02, 0.000000e+00, 0.000000e+00,
          0.000000e+00, 0.000000e+00, -2.127223e-01, 1.186893e-01, 0.000000e+00, 0.000000e+00,
          1.000000e+00, 0.000000e+00, 1.000000e+00, -2.014148e-01, 2.209044e-01, 2.687383e-02,
          1.000000e+00, -2.568389e-02, 2.108086e-01, 6.878037e-02, -2.008218e-01, 3.752568e-02,
          -2.414176e-01, -1.413018e-01, 1.705768e-01, -4.762509e-02, 1.000000e+00, 1.943306e-02,
          1.809571e-01, -2.627045e-02, 1.000000e+00, -1.559841e-02, 1.000000e+00, 0.000000e+00,
          0.000000e+00, 0.000000e+00, -3.221432e-01, -8.796441e-02, 0.000000e+00, 0.000000e+00,
          0.000000e+00, 0.000000e+00, -3.003773e-02, -1.087609e-02, -1.023158e-01, -9.410034e-02,
          -1.142110e-01, -4.599828e-02, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
          7.111933e-02, -1.348906e-02, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00 };
    for(auto& tGoldVel : tGoldVelocity)
    {
        auto tDof = &tGoldVel - &tGoldVelocity[0];
        TEST_FLOATING_EQUALITY(tGoldVel, tHostVelocity(tDof), tTol); 
    }
    //Plato::print(tVelSubView, "steady state velocity");
}


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, LidDrivenCavity_Re400)
{
    // set xml file inputs
    Teuchos::RCP<Teuchos::ParameterList> tInputs =
        Teuchos::getParametersFromXmlString(
            "<ParameterList name='Plato Problem'>"
            "  <ParameterList name='Hyperbolic'>"
            "    <Parameter name='Heat Transfer' type='string' value='None'/>"
            "    <ParameterList  name='Dimensionless Properties'>"
            "      <Parameter  name='Prandtl Number'   type='double'  value='1.7'/>"
            "      <Parameter  name='Reynolds Number'  type='double'  value='4e2'/>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList name='Spatial Model'>"
            "    <ParameterList name='Domains'>"
            "      <ParameterList name='Design Volume'>"
            "        <Parameter name='Element Block' type='string' value='body'/>"
            "        <Parameter name='Material Model' type='string' value='Water'/>"
            "      </ParameterList>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList name='Material Models'>"
            "    <ParameterList name='Water'>"
            "      <ParameterList name='Thermal Properties'>"
            "        <Parameter  name='Thermal Conductivity'  type='double'  value='1'/>"
            "      </ParameterList>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList  name='Velocity Essential Boundary Conditions'>"
            "    <ParameterList  name='X-Dir No-Slip on X-'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='0'/>"
            "      <Parameter  name='Sides'    type='string' value='x-'/>"
            "    </ParameterList>"
            "    <ParameterList  name='Y-Dir No-Slip on X-'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='1'/>"
            "      <Parameter  name='Sides'    type='string' value='x-'/>"
            "    </ParameterList>"
            "    <ParameterList  name='X-Dir No-Slip on X+'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='0'/>"
            "      <Parameter  name='Sides'    type='string' value='x+'/>"
            "    </ParameterList>"
            "    <ParameterList  name='Y-Dir No-Slip on X+'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='1'/>"
            "      <Parameter  name='Sides'    type='string' value='x+'/>"
            "    </ParameterList>"
            "    <ParameterList  name='X-Dir No-Slip on Y-'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='0'/>"
            "      <Parameter  name='Sides'    type='string' value='y-'/>"
            "    </ParameterList>"
            "    <ParameterList  name='Y-Dir No-Slip on Y-'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='1'/>"
            "      <Parameter  name='Sides'    type='string' value='y-'/>"
            "    </ParameterList>"
            "    <ParameterList  name='Tangential Velocity'>"
            "      <Parameter  name='Type'     type='string' value='Fixed Value'/>"
            "      <Parameter  name='Value'    type='double' value='1.0'/>"
            "      <Parameter  name='Index'    type='int'    value='0'/>"
            "      <Parameter  name='Sides'    type='string' value='y+'/>"
            "    </ParameterList>"
            "    <ParameterList  name='Normal Velocity'>"
            "      <Parameter  name='Type'     type='string' value='Fixed Value'/>"
            "      <Parameter  name='Value'    type='double' value='0'/>"
            "      <Parameter  name='Index'    type='int'    value='1'/>"
            "      <Parameter  name='Sides'    type='string' value='y-'/>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList  name='Pressure Essential Boundary Conditions'>"
            "    <ParameterList  name='Zero Pressure'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='0'/>"
            "      <Parameter  name='Sides'    type='string' value='pressure'/>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList  name='Newton Iteration'>"
            "    <Parameter name='Pressure Tolerance'  type='double' value='1e-4'/>"
            "    <Parameter name='Predictor Tolerance' type='double' value='1e-4'/>"
            "    <Parameter name='Corrector Tolerance' type='double' value='1e-4'/>"
            "  </ParameterList>"
            "  <ParameterList  name='Time Integration'>"
            "    <Parameter name='Safety Factor'      type='double' value='0.9'/>"
            "  </ParameterList>"
            "  <ParameterList  name='Linear Solver'>"
            "    <Parameter name='Solver Stack' type='string' value='Epetra'/>"
            "  </ParameterList>"
            "  <ParameterList  name='Convergence'>"
            "    <Parameter name='Steady State Tolerance' type='double' value='1e-7'/>"
            "  </ParameterList>"
            "</ParameterList>"
            );

    // build mesh, spatial domain, and spatial model
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,5,5);
    auto tMeshSets = PlatoUtestHelpers::get_box_mesh_sets(tMesh.operator*());
    Plato::SpatialDomain tDomain(tMesh.operator*(), tMeshSets, "box");
    tDomain.cellOrdinals("body");

    // add pressure essential boundary condition to node set list
    Omega_h::Write<int> tWritePress(1);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, 1), LAMBDA_EXPRESSION(const Plato::OrdinalType & aOrdinal)
    {
        tWritePress[aOrdinal]=0;
    }, "set pressure bc dofs value");
    auto tPressBcNodeIds = Omega_h::LOs(tWritePress);
    tMeshSets[Omega_h::NODE_SET].insert( std::pair<std::string,Omega_h::LOs>("pressure",tPressBcNodeIds) );

    // create communicator
    MPI_Comm tMyComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &tMyComm);
    Plato::Comm::Machine tMachine(tMyComm);

    // create and run incompressible cfd problem
    constexpr auto tSpaceDim = 2;
    Plato::Fluids::QuasiImplicit<Plato::IncompressibleFluids<tSpaceDim>> tProblem(*tMesh, tMeshSets, *tInputs, tMachine);
    const auto tNumVerts = tMesh->nverts();
    auto tControls = Plato::ScalarVector("Controls", tNumVerts);
    Plato::blas1::fill(1.0, tControls);
    auto tSolution = tProblem.solution(tControls);
    //tProblem.output("cfd_test_problem");

    // test solution
    auto tTags = tSolution.tags();
    std::vector<std::string> tGoldTags = { "velocity", "pressure" };
    TEST_EQUALITY(tGoldTags.size(), tTags.size());
    for(auto& tTag : tTags)
    {
        auto tItr = std::find(tGoldTags.begin(), tGoldTags.end(), tTag);
        TEST_ASSERT(tItr != tGoldTags.end());
        TEST_EQUALITY(*tItr, tTag);
    }

    auto tTol = 1e-4;
    auto tPressure = tSolution.get("pressure");
    auto tPressSubView = Kokkos::subview(tPressure, 1, Kokkos::ALL());
    auto tHostPressure = Kokkos::create_mirror(tPressSubView);
    Kokkos::deep_copy(tHostPressure, tPressSubView);
    std::vector<double> tGoldPressure = 
        { 0.000000e+00, 1.072931e-02, 1.312595e-02, 8.497650e-03, 1.626071e-02, 1.571180e-02,
          1.226569e-02, 2.217873e-02, 3.655916e-02, -1.156675e-02, -8.006286e-03, -2.601345e-01,
          -5.513425e-01, -3.071730e-01, -9.552018e-02, -1.242182e-01, -3.153474e-02, -1.812273e-02,
          -1.578797e-03, 4.713658e-03, -6.986284e-03, 8.882305e-02, 1.277841e-01, 3.863289e-01,
          1.952344e-01, 1.181027e-02, 2.533018e-02, 2.192144e-02, -2.588920e-03, 4.289334e-03,
          4.061401e-03, 6.653193e-04, 6.831889e-03, 3.598375e-03, 2.037281e-02, 1.476802e-02 };
    for(auto& tGoldPress : tGoldPressure)
    {
        auto tDof = &tGoldPress - &tGoldPressure[0];
        TEST_FLOATING_EQUALITY(tGoldPress, tHostPressure(tDof), tTol);
    }
    //Plato::print(tPressSubView, "steady state pressure");

    auto tVelocity = tSolution.get("velocity");
    auto tVelSubView = Kokkos::subview(tVelocity, 1, Kokkos::ALL());
    auto tHostVelocity = Kokkos::create_mirror(tVelSubView);
    Kokkos::deep_copy(tHostVelocity, tVelSubView);
    std::vector<double> tGoldVelocity = 
        { 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 8.662285e-02, -4.020398e-03,
          0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 1.099406e-01, 2.509727e-02,
          -2.532188e-01, -3.125290e-02, -9.557318e-02, 2.094245e-02, 0.000000e+00, 0.000000e+00,
          0.000000e+00, 0.000000e+00, -3.807676e-01, 1.177750e-01, 0.000000e+00, 0.000000e+00,
          1.000000e+00, 0.000000e+00, 1.000000e+00, -4.856890e-01, 1.468180e-01, 5.108404e-02,
          1.000000e+00, -2.508263e-01, 8.798423e-02, 7.705212e-02, -3.786300e-01, -8.369537e-02,
          -3.872812e-01, -1.725049e-01, 8.478907e-02, -2.154752e-02, 1.000000e+00, -6.767520e-02,
          3.886721e-02, -4.101624e-02, 1.000000e+00, -7.003546e-02, 1.000000e+00, 0.000000e+00,
          0.000000e+00, 0.000000e+00, -4.480170e-01, -7.927017e-02, 0.000000e+00, 0.000000e+00,
          0.000000e+00, 0.000000e+00, 7.679686e-02, 3.779762e-02, -4.002098e-03, -8.364413e-03,
          -6.939600e-02, -5.700533e-03, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
          8.581847e-02, 1.703641e-02, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00 };
    for(auto& tGoldVel : tGoldVelocity)
    {
        auto tDof = &tGoldVel - &tGoldVelocity[0];
        TEST_FLOATING_EQUALITY(tGoldVel, tHostVelocity(tDof), tTol);
    }
    //Plato::print(tVelSubView, "steady state velocity");
}


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, NaturalConvectionSquareEnclosure_Ra1e3)
{
    // set xml file inputs
    Teuchos::RCP<Teuchos::ParameterList> tInputs =
        Teuchos::getParametersFromXmlString(
            "<ParameterList name='Plato Problem'>"
            "  <ParameterList name='Hyperbolic'>"
            "    <Parameter name='Heat Transfer' type='string' value='Natural'/>"
            "    <ParameterList  name='Dimensionless Properties'>"
            "      <Parameter  name='Prandtl Number'  type='double' value='0.7'/>"
            "      <Parameter  name='Rayleigh Number' type='Array(double)' value='{0,1e3}'/>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList name='Spatial Model'>"
            "    <ParameterList name='Domains'>"
            "      <ParameterList name='Design Volume'>"
            "        <Parameter name='Element Block' type='string' value='body'/>"
            "        <Parameter name='Material Model' type='string' value='Air'/>"
            "      </ParameterList>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList  name='Velocity Essential Boundary Conditions'>"
            "    <ParameterList  name='X-Dir No-Slip on X-'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='0'/>"
            "      <Parameter  name='Sides'    type='string' value='x-'/>"
            "    </ParameterList>"
            "    <ParameterList  name='Y-Dir No-Slip on X-'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='1'/>"
            "      <Parameter  name='Sides'    type='string' value='x-'/>"
            "    </ParameterList>"
            "    <ParameterList  name='X-Dir No-Slip on X+'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='0'/>"
            "      <Parameter  name='Sides'    type='string' value='x+'/>"
            "    </ParameterList>"
            "    <ParameterList  name='Y-Dir No-Slip on X+'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='1'/>"
            "      <Parameter  name='Sides'    type='string' value='x+'/>"
            "    </ParameterList>"
            "    <ParameterList  name='X-Dir No-Slip on Y-'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='0'/>"
            "      <Parameter  name='Sides'    type='string' value='y-'/>"
            "    </ParameterList>"
            "    <ParameterList  name='Y-Dir No-Slip on Y-'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='1'/>"
            "      <Parameter  name='Sides'    type='string' value='y-'/>"
            "    </ParameterList>"
            "    <ParameterList  name='X-Dir No-Slip on Y+'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='0'/>"
            "      <Parameter  name='Sides'    type='string' value='y+'/>"
            "    </ParameterList>"
            "    <ParameterList  name='Y-Dir No-Slip on Y+'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='1'/>"
            "      <Parameter  name='Sides'    type='string' value='y+'/>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList  name='Pressure Essential Boundary Conditions'>"
            "    <ParameterList  name='Zero Pressure'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='0'/>"
            "      <Parameter  name='Sides'    type='string' value='pressure'/>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList  name='Temperature Essential Boundary Conditions'>"
            "    <ParameterList  name='Cold Wall'>"
            "      <Parameter  name='Type'     type='string' value='Fixed Value'/>"
            "      <Parameter  name='Value'    type='double' value='1.0'/>"
            "      <Parameter  name='Index'    type='int'    value='0'/>"
            "      <Parameter  name='Sides'    type='string' value='x-'/>"
            "    </ParameterList>"
            "    <ParameterList  name='Hot Wall'>"
            "      <Parameter  name='Type'     type='string' value='Fixed Value'/>"
            "      <Parameter  name='Value'    type='double' value='0.0'/>"
            "      <Parameter  name='Index'    type='int'    value='0'/>"
            "      <Parameter  name='Sides'    type='string' value='x+'/>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList  name='Newton Iteration'>"
            "    <Parameter name='Pressure Tolerance'  type='double' value='1e-4'/>"
            "    <Parameter name='Predictor Tolerance' type='double' value='1e-4'/>"
            "    <Parameter name='Corrector Tolerance' type='double' value='1e-4'/>"
            "    <Parameter name='Temperature Tolerance' type='double' value='1e-2'/>"
            "  </ParameterList>"
            "  <ParameterList  name='Time Integration'>"
            "    <Parameter name='Safety Factor'      type='double' value='1.0'/>"
            "  </ParameterList>"
            "  <ParameterList  name='Linear Solver'>"
            "    <Parameter name='Solver Stack' type='string' value='Epetra'/>"
            "  </ParameterList>"
            "  <ParameterList  name='Convergence'>"
            "    <Parameter name='Steady State Tolerance' type='double' value='1e-4'/>"
            "  </ParameterList>"
            "</ParameterList>"
            );

    // build mesh, spatial domain, and spatial model
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,20,20);
    auto tMeshSets = PlatoUtestHelpers::get_box_mesh_sets(tMesh.operator*());
    Plato::SpatialDomain tDomain(tMesh.operator*(), tMeshSets, "box");
    tDomain.cellOrdinals("body");

    // add pressure essential boundary condition to node set list
    Omega_h::Write<int> tWritePress(1);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, 1), LAMBDA_EXPRESSION(const Plato::OrdinalType & aOrdinal)
    {
        tWritePress[aOrdinal]=0;
    }, "set pressure bc dofs value");
    auto tPressBcNodeIds = Omega_h::LOs(tWritePress);
    tMeshSets[Omega_h::NODE_SET].insert( std::pair<std::string,Omega_h::LOs>("pressure",tPressBcNodeIds) );

    // create communicator
    MPI_Comm tMyComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &tMyComm);
    Plato::Comm::Machine tMachine(tMyComm);

    // create and run incompressible cfd problem
    constexpr auto tSpaceDim = 2;
    Plato::Fluids::QuasiImplicit<Plato::IncompressibleFluids<tSpaceDim>> tProblem(*tMesh, tMeshSets, *tInputs, tMachine);
    const auto tNumVerts = tMesh->nverts();
    auto tControls = Plato::ScalarVector("Controls", tNumVerts);
    Plato::blas1::fill(1.0, tControls);
    auto tSolution = tProblem.solution(tControls);
    //tProblem.output("cfd_test_problem");

    // test solution
    auto tTags = tSolution.tags();
    std::vector<std::string> tGoldTags = { "velocity", "pressure", "temperature" };
    TEST_ASSERT(tTags.size() == tGoldTags.size());
    TEST_EQUALITY(tGoldTags.size(), tTags.size());
    for(auto& tTag : tTags)
    {
        auto tItr = std::find(tGoldTags.begin(), tGoldTags.end(), tTag);
        TEST_ASSERT(tItr != tGoldTags.end());
        TEST_EQUALITY(*tItr, tTag);
    }

    auto tTol = 1e-2;
    auto tPressure = tSolution.get("pressure");
    auto tPressSubView = Kokkos::subview(tPressure, 1, Kokkos::ALL());
    Plato::Scalar tMaxPress = 0;
    Plato::blas1::max(tPressSubView, tMaxPress);
    TEST_FLOATING_EQUALITY(426.298, tMaxPress, tTol);
    Plato::Scalar tMinPress = 0;
    Plato::blas1::min(tPressSubView, tMinPress);
    TEST_FLOATING_EQUALITY(-2.11287, tMinPress, tTol);
    //Plato::print(tPressSubView, "steady state pressure");

    auto tVelocity = tSolution.get("velocity");
    auto tVelSubView = Kokkos::subview(tVelocity, 1, Kokkos::ALL());
    Plato::Scalar tMaxVel = 0;
    Plato::blas1::max(tVelSubView, tMaxVel);
    TEST_FLOATING_EQUALITY(4.90872, tMaxVel, tTol);
    Plato::Scalar tMinVel = 0;
    Plato::blas1::min(tVelSubView, tMinVel);
    TEST_FLOATING_EQUALITY(-3.43327, tMinVel, tTol);
    //Plato::print(tVelSubView, "steady state velocity");

    auto tTemperature = tSolution.get("temperature");
    auto tTempSubView = Kokkos::subview(tTemperature, 1, Kokkos::ALL());
    auto tTempNorm = Plato::blas1::norm(tTempSubView);
    TEST_FLOATING_EQUALITY(11.8231, tTempNorm, tTol);
    //Plato::print(tTempSubView, "steady state temperature");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, CalculateMisfitEuclideanNorm)
{
    // set current pressure
    constexpr auto tNumNodes = 4;
    Plato::ScalarVector tCurPressure("current pressure", tNumNodes);
    auto tHostCurPressure = Kokkos::create_mirror(tCurPressure);
    tHostCurPressure(0) = 1;
    tHostCurPressure(1) = 2;
    tHostCurPressure(2) = 3;
    tHostCurPressure(3) = 4;
    Kokkos::deep_copy(tCurPressure, tHostCurPressure);

    // set previous pressure
    Plato::ScalarVector tPrevPressure("previous pressure", tNumNodes);
    auto tHostPrevPressure = Kokkos::create_mirror(tPrevPressure);
    tHostPrevPressure(0) = 0.5;
    tHostPrevPressure(1) = 0.6;
    tHostPrevPressure(2) = 0.7;
    tHostPrevPressure(3) = 0.8;
    Kokkos::deep_copy(tPrevPressure, tHostPrevPressure);

    // call function
    constexpr auto tDofsPerNode = 1;
    auto tValue = Plato::cbs::calculate_misfit_euclidean_norm<tDofsPerNode>(tNumNodes, tCurPressure, tPrevPressure);

    // test result
    auto tTol = 1e-4;
    // TODO: FIX DUE TO CRITICAL TIME STEP CHANGES
    TEST_FLOATING_EQUALITY(5.3887059e2, tValue, tTol);
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, CalculateMisfitInfNorm)
{
    // set current pressure
    constexpr auto tNumNodes = 4;
    Plato::ScalarVector tCurPressure("current pressure", tNumNodes);
    auto tHostCurPressure = Kokkos::create_mirror(tCurPressure);
    tHostCurPressure(0) = 1;
    tHostCurPressure(1) = 2;
    tHostCurPressure(2) = 3;
    tHostCurPressure(3) = 4;
    Kokkos::deep_copy(tCurPressure, tHostCurPressure);

    // set previous pressure
    Plato::ScalarVector tPrevPressure("previous pressure", tNumNodes);
    auto tHostPrevPressure = Kokkos::create_mirror(tPrevPressure);
    tHostPrevPressure(0) = 0.5;
    tHostPrevPressure(1) = 0.6;
    tHostPrevPressure(2) = 0.7;
    tHostPrevPressure(3) = 0.8;
    Kokkos::deep_copy(tPrevPressure, tHostPrevPressure);

    // call funciton
    constexpr auto tDofsPerNode = 1;
    auto tValue = Plato::cbs::calculate_misfit_inf_norm<tDofsPerNode>(tNumNodes, tCurPressure, tPrevPressure);

    // test result
    // TODO: FIX DUE TO CRITICAL TIME STEP CHANGES
    auto tTol = 1e-4;
    TEST_FLOATING_EQUALITY(500.0, tValue, tTol);
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, CalculatePressureResidual)
{
    // set element characteristic size
    constexpr auto tNumNodes = 4;
    Plato::ScalarVector tCurPressure("current pressure", tNumNodes);
    auto tHostCurPressure = Kokkos::create_mirror(tCurPressure);
    tHostCurPressure(0) = 1;
    tHostCurPressure(1) = 2;
    tHostCurPressure(2) = 3;
    tHostCurPressure(3) = 4;
    Kokkos::deep_copy(tCurPressure, tHostCurPressure);

    // set convective velocity
    Plato::ScalarVector tPrevPressure("previous pressure", tNumNodes);
    auto tHostPrevPressure = Kokkos::create_mirror(tPrevPressure);
    tHostPrevPressure(0) = 0.5;
    tHostPrevPressure(1) = 0.6;
    tHostPrevPressure(2) = 0.7;
    tHostPrevPressure(3) = 0.8;
    Kokkos::deep_copy(tPrevPressure, tHostPrevPressure);

    // call function
    constexpr auto tDofsPerNode = 1;
    auto tResidual = Plato::cbs::calculate_field_misfit<tDofsPerNode>(tNumNodes, tCurPressure, tPrevPressure);

    // test results
    // TODO: FIX DUE TO CRITICAL TIME STEP CHANGES
    auto tTol = 1e-4;
    auto tHostResidual = Kokkos::create_mirror(tResidual);
    Kokkos::deep_copy(tHostResidual, tResidual);
    std::vector<Plato::Scalar> tGold = {500,175,85.185185185,50};
    for (Plato::OrdinalType tNode = 0; tNode < tNumNodes; tNode++)
    {
        TEST_FLOATING_EQUALITY(tGold[tNode], tHostResidual(tNode), tTol); // @suppress("Invalid arguments")
    }
    //Plato::print(tResidual, "residual");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, CalculateThermalVelocityMagnitude)
{
    // build mesh and spatial model
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);
    auto tMeshSets = PlatoUtestHelpers::get_box_mesh_sets(tMesh.operator*());
    Plato::SpatialModel tSpatialModel(tMesh.operator*(), tMeshSets);

    // set element characteristic size
    auto tNumNodes = tMesh->nverts();
    Plato::ScalarVector tElemCharSize("char elem size", tNumNodes);
    auto tHostElemCharSize = Kokkos::create_mirror(tElemCharSize);
    tHostElemCharSize(0) = 0.5;
    tHostElemCharSize(1) = 0.5;
    tHostElemCharSize(2) = 1.0;
    tHostElemCharSize(3) = 1.0;
    Kokkos::deep_copy(tElemCharSize, tHostElemCharSize);

    // call function
    auto tPrandtlNum = 1;
    auto tReynoldsNum = 2;
    constexpr auto tNumNodesPerCell = 3;
    auto tThermalVelocity =
        Plato::cbs::calculate_thermal_velocity_magnitude<tNumNodesPerCell>(tSpatialModel, tPrandtlNum, tReynoldsNum, tElemCharSize);

    // test value
    auto tTol = 1e-4;
    auto tHostThermalVelocity = Kokkos::create_mirror(tThermalVelocity);
    Kokkos::deep_copy(tHostThermalVelocity, tThermalVelocity);
    std::vector<Plato::Scalar> tGold = {1.0,1.0,0.5,0.5};
    for (decltype(tNumNodes) tNode = 0; tNode < tNumNodes; tNode++)
    {
        TEST_FLOATING_EQUALITY(tGold[tNode], tHostThermalVelocity(tNode), tTol);
    }
    //Plato::print(tThermalVelocity, "thermal velocity");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, CalculateDiffusiveVelocityMagnitude)
{
    // build mesh and spatial model
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);
    auto tMeshSets = PlatoUtestHelpers::get_box_mesh_sets(tMesh.operator*());
    Plato::SpatialModel tSpatialModel(tMesh.operator*(), tMeshSets);

    // set element characteristic size
    auto tNumNodes = tMesh->nverts();
    Plato::ScalarVector tElemCharSize("char elem size", tNumNodes);
    auto tHostElemCharSize = Kokkos::create_mirror(tElemCharSize);
    tHostElemCharSize(0) = 0.5;
    tHostElemCharSize(1) = 0.5;
    tHostElemCharSize(2) = 1.0;
    tHostElemCharSize(3) = 1.0;
    Kokkos::deep_copy(tElemCharSize, tHostElemCharSize);

    // call function
    auto tReynoldsNum = 2;
    constexpr auto tNumNodesPerCell = 3;
    auto tDiffusiveVelocity =
        Plato::cbs::calculate_diffusive_velocity_magnitude<tNumNodesPerCell>(tSpatialModel, tReynoldsNum, tElemCharSize);

    // test value
    auto tTol = 1e-4;
    auto tHostDiffusiveVelocity = Kokkos::create_mirror(tDiffusiveVelocity);
    Kokkos::deep_copy(tHostDiffusiveVelocity, tDiffusiveVelocity);
    std::vector<Plato::Scalar> tGold = {1.0,1.0,0.5,0.5};
    for (decltype(tNumNodes) tNode = 0; tNode < tNumNodes; tNode++)
    {
        TEST_FLOATING_EQUALITY(tGold[tNode], tHostDiffusiveVelocity(tNode), tTol);
    }
    //Plato::print(tDiffusiveVelocity, "diffusive velocity");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, CalculateConvectiveVelocityMagnitude)
{
    // build mesh and spatial model
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);
    auto tMeshSets = PlatoUtestHelpers::get_box_mesh_sets(tMesh.operator*());
    Plato::SpatialModel tSpatialModel(tMesh.operator*(), tMeshSets);

    // set velocity field
    auto tNumNodes = tMesh->nverts();
    auto tNumSpaceDims = tMesh->dim();
    Plato::ScalarVector tVelocity("velocity", tNumNodes * tNumSpaceDims);
    auto tHostVelocity = Kokkos::create_mirror(tVelocity);
    tHostVelocity(0) = 1;
    tHostVelocity(1) = 2;
    tHostVelocity(2) = 3;
    tHostVelocity(3) = 4;
    tHostVelocity(4) = 5;
    tHostVelocity(5) = 6;
    tHostVelocity(6) = 7;
    tHostVelocity(7) = 8;
    Kokkos::deep_copy(tVelocity, tHostVelocity);

    // call function
    constexpr auto tNumNodesPerCell = 3;
    auto tConvectiveVelocity =
        Plato::cbs::calculate_convective_velocity_magnitude<tNumNodesPerCell>(tSpatialModel, tVelocity);

    // test value
    auto tTol = 1e-4;
    auto tHostConvectiveVelocity = Kokkos::create_mirror(tConvectiveVelocity);
    Kokkos::deep_copy(tHostConvectiveVelocity, tConvectiveVelocity);
    std::vector<Plato::Scalar> tGold = {2.23606797749978969640,5.0,7.81024967590665439412,10.63014581273464940799};
    for (decltype(tNumNodes) tNode = 0; tNode < tNumNodes; tNode++)
    {
        TEST_FLOATING_EQUALITY(tGold[tNode], tHostConvectiveVelocity(tNode), tTol);
    }
    //Plato::print(tConvectiveVelocity, "convective velocity");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, CalculateArtificialCompressibility)
{
    constexpr auto tNumNodes = 4;
    Plato::ScalarVector tConvectiveVelocity("convective velocity", tNumNodes);
    auto tHostConvectiveVelocity = Kokkos::create_mirror(tConvectiveVelocity);
    tHostConvectiveVelocity(0) = 1.0;
    tHostConvectiveVelocity(1) = 2.0;
    tHostConvectiveVelocity(2) = 3.0;
    tHostConvectiveVelocity(3) = 0.3;
    Kokkos::deep_copy(tConvectiveVelocity, tHostConvectiveVelocity);
    Plato::ScalarVector tDiffusiveVelocity("diffusive velocity", tNumNodes);
    auto tHostDiffusiveVelocity = Kokkos::create_mirror(tDiffusiveVelocity);
    tHostDiffusiveVelocity(0) = 0.9;
    tHostDiffusiveVelocity(1) = 3.0;
    tHostDiffusiveVelocity(2) = 2.0;
    tHostDiffusiveVelocity(3) = 0.2;
    Kokkos::deep_copy(tDiffusiveVelocity, tHostDiffusiveVelocity);
    Plato::ScalarVector tThermalVelocity("thermal velocity", tNumNodes);
    auto tHostThermalVelocity = Kokkos::create_mirror(tThermalVelocity);
    tHostThermalVelocity(0) = 0.9;
    tHostThermalVelocity(1) = 3.0;
    tHostThermalVelocity(2) = 4.0;
    tHostThermalVelocity(3) = 0.1;
    Kokkos::deep_copy(tThermalVelocity, tHostThermalVelocity);
    auto tCriticalValue = 0.5;

    auto tOutput =
        Plato::cbs::calculate_artificial_compressibility(tConvectiveVelocity, tDiffusiveVelocity, tThermalVelocity, tCriticalValue);

    // test value
    auto tTol = 1e-4;
    auto tHostOutput = Kokkos::create_mirror(tOutput);
    Kokkos::deep_copy(tHostOutput, tOutput);
    std::vector<Plato::Scalar> tGold = {1.0,3.0,4.0,0.5};
    for (Plato::OrdinalType tNode = 0; tNode < tNumNodes; tNode++)
    {
        TEST_FLOATING_EQUALITY(tGold[tNode], tHostOutput(tNode), tTol);
    }
    //Plato::print(tOutput, "artificial compressibility");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, CalculateElementCharacteristicSizes)
{
    // build mesh and spatial model
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);
    auto tMeshSets = PlatoUtestHelpers::get_box_mesh_sets(tMesh.operator*());
    Plato::SpatialModel tSpatialModel(tMesh.operator*(), tMeshSets);

    constexpr auto tNumSpaceDims = 2;
    constexpr auto tNumNodesPerCell = tNumSpaceDims + 1;
    auto tElemCharSize =
        Plato::cbs::calculate_element_characteristic_sizes<tNumSpaceDims,tNumNodesPerCell>(tSpatialModel);

    // test value
    auto tTol = 1e-4;
    auto tHostElemCharSize = Kokkos::create_mirror(tElemCharSize);
    Kokkos::deep_copy(tHostElemCharSize, tElemCharSize);
    std::vector<Plato::Scalar> tGold = {5.857864e-01,5.857864e-01,5.857864e-01,5.857864e-01};

    auto tNumNodes = tSpatialModel.Mesh.nverts();
    for (Plato::OrdinalType tNode = 0; tNode < tNumNodes; tNode++)
    {
        TEST_FLOATING_EQUALITY(tGold[tNode], tHostElemCharSize(tNode), tTol);
    }
    //Plato::print(tElemCharSize, "element characteristic size");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, TemperatureIncrementResidual_EvaluatePrescribed)
{
    // set xml file inputs
    Teuchos::RCP<Teuchos::ParameterList> tInputs =
        Teuchos::getParametersFromXmlString(
            "<ParameterList name='Plato Problem'>"
            "  <ParameterList name='Hyperbolic'>"
            "    <Parameter name='Heat Transfer' type='string' value='Natural'/>"
            "    <ParameterList  name='Dimensionless Properties'>"
            "      <Parameter  name='Characteristic Length' type='double' value='1.0'/>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList name='Material Models'>"
            "    <ParameterList name='Madeuptinum'>"
            "      <ParameterList name='Thermal Properties'>"
            "        <Parameter  name='Thermal Conductivity'  type='double'  value='1'/>"
            "        <Parameter  name='Reference Temperature'       type='double'  value='10.0'/>"
            "        <Parameter  name='Thermal Diffusivity Ratio'   type='double'  value='1.0'/>"
            "      </ParameterList>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList  name='Energy Natural Boundary Conditions'>"
            "    <ParameterList  name='Heat Flux'>"
            "      <Parameter  name='Type'   type='string'  value='Uniform'/>"
            "      <Parameter  name='Sides'  type='string'  value='x+'/>"
            "      <Parameter  name='Value'  type='double'  value='-1.0'/>"
            "    </ParameterList>"
            "  </ParameterList>"
            "</ParameterList>"
            );

    // set physics and evaluation type
    constexpr Plato::OrdinalType tNumSpaceDims = 2;
    using PhysicsT = Plato::IncompressibleFluids<tNumSpaceDims>::EnergyPhysicsT;
    using EvaluationT = Plato::Fluids::Evaluation<PhysicsT::SimplexT>::Residual;

    // build mesh, spatial domain, and spatial model
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);
    auto tMeshSets = PlatoUtestHelpers::get_box_mesh_sets(tMesh.operator*());
    Plato::SpatialDomain tDomain(tMesh.operator*(), tMeshSets, "box");
    tDomain.cellOrdinals("body");
    tDomain.setMaterialName("Madeuptinum");
    tDomain.setElementBlockName("block_1");
    Plato::SpatialModel tSpatialModel(tMesh.operator*(), tMeshSets);
    tSpatialModel.append(tDomain);

    // set workset
    Plato::WorkSets tWorkSets;
    auto tNumCells = tMesh->nelems();
    constexpr auto tNumNodesPerCell = tNumSpaceDims + 1;
    using ConfigT = EvaluationT::ConfigScalarType;
    Plato::NodeCoordinate<tNumSpaceDims> tNodeCoordinate( (&tMesh.operator*()) );
    auto tConfig = std::make_shared< Plato::MetaData< Plato::ScalarArray3DT<ConfigT> > >
        ( Plato::ScalarArray3DT<ConfigT>("configuration", tNumCells, tNumNodesPerCell, tNumSpaceDims) );
    Plato::workset_config_scalar<tNumSpaceDims, tNumNodesPerCell>(tMesh->nelems(), tNodeCoordinate, tConfig->mData);
    tWorkSets.set("configuration", tConfig);

    using ControlT = EvaluationT::ControlScalarType;
    auto tControl = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<ControlT> > >
        ( Plato::ScalarMultiVectorT<ControlT>("control", tNumCells, PhysicsT::mNumControlDofsPerCell) );
    Plato::blas2::fill(1.0, tControl->mData);
    tWorkSets.set("control", tControl);

    using PrevTempT = EvaluationT::PreviousMassScalarType;
    auto tPrevTemp = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<PrevTempT> > >
        ( Plato::ScalarMultiVectorT<PrevTempT>("previous pressure", tNumCells, PhysicsT::mNumEnergyDofsPerCell) );
    auto tHostPrevTemp = Kokkos::create_mirror(tPrevTemp->mData);
    tHostPrevTemp(0, 0) = 1; tHostPrevTemp(1, 0) = 4;
    tHostPrevTemp(0, 1) = 2; tHostPrevTemp(1, 1) = 5;
    tHostPrevTemp(0, 2) = 3; tHostPrevTemp(1, 2) = 6;
    Kokkos::deep_copy(tPrevTemp->mData, tHostPrevTemp);
    tWorkSets.set("previous temperature", tPrevTemp);

    auto tTimeStep = std::make_shared< Plato::MetaData< Plato::ScalarVector > >( Plato::ScalarVector("time step", 1) );
    auto tHostTimeStep = Kokkos::create_mirror(tTimeStep->mData);
    tHostTimeStep(0) = 0.01;
    Kokkos::deep_copy(tTimeStep->mData, tHostTimeStep);
    tWorkSets.set("critical time step", tTimeStep);

    // evaluate temperature increment residual
    Plato::DataMap tDataMap;
    Plato::ScalarMultiVectorT<EvaluationT::ResultScalarType> tResult("result", tNumCells, PhysicsT::mNumEnergyDofsPerCell);
    Plato::Fluids::SIMP::TemperatureResidual<PhysicsT,EvaluationT> tResidual(tDomain,tDataMap,tInputs.operator*());
    tResidual.evaluatePrescribed(tSpatialModel, tWorkSets, tResult);

    // test values
    // TODO: FIX DUE TO CRITICAL TIME STEP CHANGES
    auto tTol = 1e-4;
    auto tHostResult = Kokkos::create_mirror(tResult);
    Kokkos::deep_copy(tHostResult, tResult);
    std::vector<std::vector<Plato::Scalar>> tGold =
        {{0.0,0.01,0.015}, {0.0,0.0,0.0}};
    for (Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++)
    {
        for (Plato::OrdinalType tDof = 0; tDof < PhysicsT::mNumEnergyDofsPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(tGold[tCell][tDof], tHostResult(tCell, tDof), tTol);
        }
    }
    //Plato::print_array_2D(tResult, "results");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, Brinkman_VelocityPredictorResidual_EvaluateBoundary)
{
    // set xml file inputs
    Teuchos::RCP<Teuchos::ParameterList> tInputs =
        Teuchos::getParametersFromXmlString(
            "<ParameterList name='Plato Problem'>"
            "  <ParameterList name='Hyperbolic'>"
            "    <Parameter name='Heat Transfer' type='string' value='Natural'/>"
            "    <ParameterList  name='Dimensionless Properties'>"
            "      <Parameter  name='Darcy Number'   type='double'        value='1.0'/>"
            "      <Parameter  name='Prandtl Number' type='double'        value='1.0'/>"
            "      <Parameter  name='Grashof Number' type='Array(double)' value='{0.0,1.0}'/>"
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
    using PhysicsT = Plato::IncompressibleFluids<tNumSpaceDims>::MomentumPhysicsT;
    using EvaluationT = Plato::Fluids::Evaluation<PhysicsT::SimplexT>::Residual;

    // build mesh, spatial domain, and spatial model
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);
    auto tMeshSets = PlatoUtestHelpers::get_box_mesh_sets(tMesh.operator*());
    auto tSpatialDomainName = std::string("my box");
    Plato::SpatialDomain tDomain(tMesh.operator*(), tMeshSets, tSpatialDomainName);
    auto tElementBlockName = std::string("body");
    tDomain.cellOrdinals(tElementBlockName);
    Plato::SpatialModel tSpatialModel(tMesh.operator*(), tMeshSets);
    tSpatialModel.append(tDomain);

    // set workset
    Plato::WorkSets tWorkSets;
    auto tNumCells = tMesh->nelems();
    constexpr auto tNumNodesPerCell = tNumSpaceDims + 1;
    using ConfigT = EvaluationT::ConfigScalarType;
    Plato::NodeCoordinate<tNumSpaceDims> tNodeCoordinate( (&tMesh.operator*()) );
    auto tConfig = std::make_shared< Plato::MetaData< Plato::ScalarArray3DT<ConfigT> > >
        ( Plato::ScalarArray3DT<ConfigT>("configuration", tNumCells, tNumNodesPerCell, tNumSpaceDims) );
    Plato::workset_config_scalar<tNumSpaceDims, tNumNodesPerCell>(tMesh->nelems(), tNodeCoordinate, tConfig->mData);
    tWorkSets.set("configuration", tConfig);

    using ControlT = EvaluationT::ControlScalarType;
    auto tControl = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<ControlT> > >
        ( Plato::ScalarMultiVectorT<ControlT>("control", tNumCells, PhysicsT::mNumControlDofsPerCell) );
    Plato::blas2::fill(1.0, tControl->mData);
    tWorkSets.set("control", tControl);

    using PrevVelT = EvaluationT::PreviousMomentumScalarType;
    auto tPrevVel = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<PrevVelT> > >
        ( Plato::ScalarMultiVectorT<PrevVelT>("previous velocity", tNumCells, PhysicsT::mNumMomentumDofsPerCell) );
    auto tHostVelocity = Kokkos::create_mirror(tPrevVel->mData);
    tHostVelocity(0, 0) = 1; tHostVelocity(1, 0) = 7;
    tHostVelocity(0, 1) = 2; tHostVelocity(1, 1) = 8;
    tHostVelocity(0, 2) = 3; tHostVelocity(1, 2) = 9;
    tHostVelocity(0, 3) = 4; tHostVelocity(1, 3) = 10;
    tHostVelocity(0, 4) = 5; tHostVelocity(1, 4) = 11;
    tHostVelocity(0, 5) = 6; tHostVelocity(1, 5) = 12;
    Kokkos::deep_copy(tPrevVel->mData, tHostVelocity);
    tWorkSets.set("previous velocity", tPrevVel);

    using PrevPressT = EvaluationT::PreviousMassScalarType;
    auto tPrevPress = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<PrevPressT> > >
        ( Plato::ScalarMultiVectorT<PrevPressT>("previous pressure", tNumCells, PhysicsT::mNumMassDofsPerCell) );
    auto tHostPrevPress = Kokkos::create_mirror(tPrevPress->mData);
    tHostPrevPress(0, 0) = 1; tHostPrevPress(1, 0) = 4;
    tHostPrevPress(0, 1) = 2; tHostPrevPress(1, 1) = 5;
    tHostPrevPress(0, 2) = 3; tHostPrevPress(1, 2) = 6;
    Kokkos::deep_copy(tPrevPress->mData, tHostPrevPress);
    tWorkSets.set("previous pressure", tPrevPress);

    auto tTimeStep = std::make_shared< Plato::MetaData< Plato::ScalarVector > >( Plato::ScalarVector("time step", 1) );
    auto tHostTimeStep = Kokkos::create_mirror(tTimeStep->mData);
    tHostTimeStep(0) = 0.01;
    Kokkos::deep_copy(tTimeStep->mData, tHostTimeStep);
    tWorkSets.set("critical time step", tTimeStep);

    // evaluate pressure increment residual
    Plato::DataMap tDataMap;
    Plato::ScalarMultiVectorT<EvaluationT::ResultScalarType> tResult("result", tNumCells, PhysicsT::mNumMomentumDofsPerCell);
    Plato::Fluids::Brinkman::VelocityPredictorResidual<PhysicsT,EvaluationT> tResidual(tDomain,tDataMap,tInputs.operator*());
    tResidual.evaluateBoundary(tSpatialModel, tWorkSets, tResult);

    // test values
    // TODO: FIX DUE TO CRITICAL TIME STEP CHANGES
    auto tTol = 1e-4;
    auto tHostResult = Kokkos::create_mirror(tResult);
    Kokkos::deep_copy(tHostResult, tResult);
    std::vector<std::vector<Plato::Scalar>> tGold =
        {{0.02,0.02,0.04,0.04,0.0,0.0}, {-0.08,-0.08,-0.1,-0.1,0.0,0.0}};
    for (Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++)
    {
        for (Plato::OrdinalType tDof = 0; tDof < PhysicsT::mNumMomentumDofsPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(tGold[tCell][tDof], tHostResult(tCell, tDof), tTol);
        }
    }
    //Plato::print_array_2D(tResult, "results");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, Brinkman_VelocityPredictorResidual_EvaluatePrescribedBoundary)
{
    // set xml file inputs
    Teuchos::RCP<Teuchos::ParameterList> tInputs =
        Teuchos::getParametersFromXmlString(
            "<ParameterList name='Plato Problem'>"
            "  <ParameterList name='Hyperbolic'>"
            "    <Parameter name='Heat Transfer' type='string' value='Natural'/>"
            "    <ParameterList  name='Dimensionless Properties'>"
            "      <Parameter  name='Darcy Number'   type='double'        value='1.0'/>"
            "      <Parameter  name='Prandtl Number' type='double'        value='1.0'/>"
            "      <Parameter  name='Grashof Number' type='Array(double)' value='{0.0,1.0}'/>"
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
    using PhysicsT = Plato::IncompressibleFluids<tNumSpaceDims>::MomentumPhysicsT;
    using EvaluationT = Plato::Fluids::Evaluation<PhysicsT::SimplexT>::Residual;

    // build mesh, spatial domain, and spatial model
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);
    auto tMeshSets = PlatoUtestHelpers::get_box_mesh_sets(tMesh.operator*());
    auto tSpatialDomainName = std::string("my box");
    Plato::SpatialDomain tDomain(tMesh.operator*(), tMeshSets, tSpatialDomainName);
    auto tElementBlockName = std::string("body");
    tDomain.cellOrdinals(tElementBlockName);
    Plato::SpatialModel tSpatialModel(tMesh.operator*(), tMeshSets);
    tSpatialModel.append(tDomain);

    // set workset
    Plato::WorkSets tWorkSets;
    auto tNumCells = tMesh->nelems();
    constexpr auto tNumNodesPerCell = tNumSpaceDims + 1;
    using ConfigT = EvaluationT::ConfigScalarType;
    Plato::NodeCoordinate<tNumSpaceDims> tNodeCoordinate( (&tMesh.operator*()) );
    auto tConfig = std::make_shared< Plato::MetaData< Plato::ScalarArray3DT<ConfigT> > >
        ( Plato::ScalarArray3DT<ConfigT>("configuration", tNumCells, tNumNodesPerCell, tNumSpaceDims) );
    Plato::workset_config_scalar<tNumSpaceDims, tNumNodesPerCell>(tMesh->nelems(), tNodeCoordinate, tConfig->mData);
    tWorkSets.set("configuration", tConfig);

    using ControlT = EvaluationT::ControlScalarType;
    auto tControl = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<ControlT> > >
        ( Plato::ScalarMultiVectorT<ControlT>("control", tNumCells, PhysicsT::mNumControlDofsPerCell) );
    Plato::blas2::fill(1.0, tControl->mData);
    tWorkSets.set("control", tControl);

    using PrevVelT = EvaluationT::PreviousMomentumScalarType;
    auto tPrevVel = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<PrevVelT> > >
        ( Plato::ScalarMultiVectorT<PrevVelT>("previous velocity", tNumCells, PhysicsT::mNumMomentumDofsPerCell) );
    auto tHostVelocity = Kokkos::create_mirror(tPrevVel->mData);
    tHostVelocity(0, 0) = 1; tHostVelocity(1, 0) = 7;
    tHostVelocity(0, 1) = 2; tHostVelocity(1, 1) = 8;
    tHostVelocity(0, 2) = 3; tHostVelocity(1, 2) = 9;
    tHostVelocity(0, 3) = 4; tHostVelocity(1, 3) = 10;
    tHostVelocity(0, 4) = 5; tHostVelocity(1, 4) = 11;
    tHostVelocity(0, 5) = 6; tHostVelocity(1, 5) = 12;
    Kokkos::deep_copy(tPrevVel->mData, tHostVelocity);
    tWorkSets.set("previous velocity", tPrevVel);

    using PrevPressT = EvaluationT::PreviousMassScalarType;
    auto tPrevPress = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<PrevPressT> > >
        ( Plato::ScalarMultiVectorT<PrevPressT>("previous pressure", tNumCells, PhysicsT::mNumMassDofsPerCell) );
    auto tHostPrevPress = Kokkos::create_mirror(tPrevPress->mData);
    tHostPrevPress(0, 0) = 1; tHostPrevPress(1, 0) = 4;
    tHostPrevPress(0, 1) = 2; tHostPrevPress(1, 1) = 5;
    tHostPrevPress(0, 2) = 3; tHostPrevPress(1, 2) = 6;
    Kokkos::deep_copy(tPrevPress->mData, tHostPrevPress);
    tWorkSets.set("previous pressure", tPrevPress);

    auto tTimeStep = std::make_shared< Plato::MetaData< Plato::ScalarVector > >( Plato::ScalarVector("time step", 1) );
    auto tHostTimeStep = Kokkos::create_mirror(tTimeStep->mData);
    tHostTimeStep(0) = 0.01;
    Kokkos::deep_copy(tTimeStep->mData, tHostTimeStep);
    tWorkSets.set("critical time step", tTimeStep);

    // evaluate pressure increment residual
    Plato::DataMap tDataMap;
    Plato::ScalarMultiVectorT<EvaluationT::ResultScalarType> tResult("result", tNumCells, PhysicsT::mNumMomentumDofsPerCell);
    Plato::Fluids::Brinkman::VelocityPredictorResidual<PhysicsT,EvaluationT> tResidual(tDomain,tDataMap,tInputs.operator*());
    tResidual.evaluatePrescribed(tSpatialModel, tWorkSets, tResult);

    // test values
    // TODO: FIX DUE TO CRITICAL TIME STEP CHANGES
    auto tTol = 1e-4;
    auto tHostResult = Kokkos::create_mirror(tResult);
    Kokkos::deep_copy(tHostResult, tResult);
    std::vector<std::vector<Plato::Scalar>> tGold =
        {{0.0,0.0,-0.02,0.01,-0.03,0.015}, {0.0,0.0,0.0,0.0,0.0,0.0}};
    for (Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++)
    {
        for (Plato::OrdinalType tDof = 0; tDof < PhysicsT::mNumMomentumDofsPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(tGold[tCell][tDof], tHostResult(tCell, tDof), tTol);
        }
    }
    //Plato::print_array_2D(tResult, "results");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PressureIncrementResidual_EvaluateBoundary)
{
    // set xml file inputs
    Teuchos::RCP<Teuchos::ParameterList> tInputs =
        Teuchos::getParametersFromXmlString(
            "<ParameterList name='Plato Problem'>"
            "  <ParameterList  name='Velocity Essential Boundary Conditions'>"
            "    <ParameterList  name='Zero Velocity X-Dir'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='0'/>"
            "      <Parameter  name='Sides'    type='string' value='x-'/>"
            "    </ParameterList>"
            "    <ParameterList  name='Zero Velocity Y-Dir'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='1'/>"
            "      <Parameter  name='Sides'    type='string' value='x-'/>"
            "    </ParameterList>"
            "  </ParameterList>"
            "</ParameterList>"
            );

    // set physics and evaluation type
    constexpr Plato::OrdinalType tNumSpaceDims = 2;
    using PhysicsT = Plato::IncompressibleFluids<tNumSpaceDims>::MassPhysicsT;
    using EvaluationT = Plato::Fluids::Evaluation<PhysicsT::SimplexT>::Residual;

    // build mesh, spatial domain, and spatial model
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);
    auto tMeshSets = PlatoUtestHelpers::get_box_mesh_sets(tMesh.operator*());
    auto tSpatialDomainName = std::string("my box");
    Plato::SpatialDomain tDomain(tMesh.operator*(), tMeshSets, tSpatialDomainName);
    auto tElementBlockName = std::string("body");
    tDomain.cellOrdinals(tElementBlockName);
    Plato::SpatialModel tSpatialModel(tMesh.operator*(), tMeshSets);
    tSpatialModel.append(tDomain);

    // set workset
    Plato::WorkSets tWorkSets;
    auto tNumCells = tMesh->nelems();
    constexpr auto tNumNodesPerCell = tNumSpaceDims + 1;
    using ConfigT = EvaluationT::ConfigScalarType;
    Plato::NodeCoordinate<tNumSpaceDims> tNodeCoordinate( (&tMesh.operator*()) );
    auto tConfig = std::make_shared< Plato::MetaData< Plato::ScalarArray3DT<ConfigT> > >
        ( Plato::ScalarArray3DT<ConfigT>("configuration", tNumCells, tNumNodesPerCell, tNumSpaceDims) );
    Plato::workset_config_scalar<tNumSpaceDims, tNumNodesPerCell>(tMesh->nelems(), tNodeCoordinate, tConfig->mData);
    tWorkSets.set("configuration", tConfig);

    using PrevVelT = EvaluationT::PreviousMomentumScalarType;
    auto tPrevVel = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<PrevVelT> > >
        ( Plato::ScalarMultiVectorT<PrevVelT>("previous velocity", tNumCells, PhysicsT::mNumMomentumDofsPerCell) );
    auto tHostVelocity = Kokkos::create_mirror(tPrevVel->mData);
    tHostVelocity(0, 0) = 1; tHostVelocity(1, 0) = 11;
    tHostVelocity(0, 1) = 2; tHostVelocity(1, 1) = 12;
    tHostVelocity(0, 2) = 3; tHostVelocity(1, 2) = 13;
    tHostVelocity(0, 3) = 4; tHostVelocity(1, 3) = 14;
    tHostVelocity(0, 4) = 5; tHostVelocity(1, 4) = 15;
    tHostVelocity(0, 5) = 6; tHostVelocity(1, 5) = 16;
    Kokkos::deep_copy(tPrevVel->mData, tHostVelocity);
    tWorkSets.set("previous velocity", tPrevVel);

    auto tTimeStep = std::make_shared< Plato::MetaData< Plato::ScalarVector > >( Plato::ScalarVector("time step", 1) );
    auto tHostTimeStep = Kokkos::create_mirror(tTimeStep->mData);
    tHostTimeStep(0) = 0.01;
    Kokkos::deep_copy(tTimeStep->mData, tHostTimeStep);
    tWorkSets.set("critical time step", tTimeStep);

    // evaluate pressure increment residual
    Plato::DataMap tDataMap;
    Plato::ScalarMultiVectorT<EvaluationT::ResultScalarType> tResult("result", tNumCells, PhysicsT::mNumMassDofsPerCell);
    Plato::Fluids::PressureResidual<PhysicsT,EvaluationT> tResidual(tDomain,tDataMap,tInputs.operator*());
    tResidual.evaluateBoundary(tSpatialModel, tWorkSets, tResult);

    // test values
    // TODO: FIX DUE TO CRITICAL TIME STEP CHANGES
    auto tTol = 1e-4;
    auto tHostResult = Kokkos::create_mirror(tResult);
    Kokkos::deep_copy(tHostResult, tResult);
    std::vector<std::vector<Plato::Scalar>> tGold =
        {{0.0,0.0,0.0},{0.0,0.15125,0.1815}};
    for (Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++)
    {
        for (Plato::OrdinalType tDof = 0; tDof < tNumNodesPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(tGold[tCell][tDof], tHostResult(tCell, tDof), tTol);
        }
    }
    //Plato::print_array_2D(tResult, "results");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PressureResidual)
{
    // set physics and evaluation type
    constexpr Plato::OrdinalType tNumSpaceDims = 2;
    using PhysicsT = Plato::IncompressibleFluids<tNumSpaceDims>::MassPhysicsT;
    using EvaluationT = Plato::Fluids::Evaluation<PhysicsT::SimplexT>::Residual;

    // build mesh, spatial domain, and spatial model
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);
    auto tMeshSets = PlatoUtestHelpers::get_box_mesh_sets(tMesh.operator*());
    auto tSpatialDomainName = std::string("my box");
    Plato::SpatialDomain tDomain(tMesh.operator*(), tMeshSets, tSpatialDomainName);
    auto tElementBlockName = std::string("body");
    tDomain.cellOrdinals(tElementBlockName);

    // set workset
    Plato::WorkSets tWorkSets;
    auto tNumCells = tMesh->nelems();
    constexpr auto tNumNodesPerCell = tNumSpaceDims + 1;
    using ConfigT = EvaluationT::ConfigScalarType;
    Plato::NodeCoordinate<tNumSpaceDims> tNodeCoordinate( (&tMesh.operator*()) );
    auto tConfig = std::make_shared< Plato::MetaData< Plato::ScalarArray3DT<ConfigT> > >
        ( Plato::ScalarArray3DT<ConfigT>("configuration", tNumCells, tNumNodesPerCell, tNumSpaceDims) );
    Plato::workset_config_scalar<tNumSpaceDims, tNumNodesPerCell>(tMesh->nelems(), tNodeCoordinate, tConfig->mData);
    tWorkSets.set("configuration", tConfig);

    using PrevVelT = EvaluationT::PreviousMomentumScalarType;
    auto tPrevVel = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<PrevVelT> > >
        ( Plato::ScalarMultiVectorT<PrevVelT>("previous velocity", tNumCells, PhysicsT::mNumMomentumDofsPerCell) );
    auto tHostVelocity = Kokkos::create_mirror(tPrevVel->mData);
    tHostVelocity(0, 0) = 1; tHostVelocity(1, 0) = 7;
    tHostVelocity(0, 1) = 2; tHostVelocity(1, 1) = 8;
    tHostVelocity(0, 2) = 3; tHostVelocity(1, 2) = 9;
    tHostVelocity(0, 3) = 4; tHostVelocity(1, 3) = 10;
    tHostVelocity(0, 4) = 5; tHostVelocity(1, 4) = 11;
    tHostVelocity(0, 5) = 6; tHostVelocity(1, 5) = 12;
    Kokkos::deep_copy(tPrevVel->mData, tHostVelocity);
    tWorkSets.set("previous velocity", tPrevVel);

    using PredictorT = EvaluationT::MomentumPredictorScalarType;
    auto tPredictor = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<PredictorT> > >
        ( Plato::ScalarMultiVectorT<PredictorT>("predictor", tNumCells, PhysicsT::mNumMomentumDofsPerCell) );
    auto tHostPredictor = Kokkos::create_mirror(tPredictor->mData);
    tHostPredictor(0, 0) = 0.1; tHostPredictor(1, 0) = 0.7;
    tHostPredictor(0, 1) = 0.2; tHostPredictor(1, 1) = 0.8;
    tHostPredictor(0, 2) = 0.3; tHostPredictor(1, 2) = 0.9;
    tHostPredictor(0, 3) = 0.4; tHostPredictor(1, 3) = 1.0;
    tHostPredictor(0, 4) = 0.5; tHostPredictor(1, 4) = 1.1;
    tHostPredictor(0, 5) = 0.6; tHostPredictor(1, 5) = 1.2;
    Kokkos::deep_copy(tPredictor->mData, tHostPredictor);
    tWorkSets.set("current predictor", tPredictor);

    using PrevPressT = EvaluationT::PreviousMassScalarType;
    auto tPrevPress = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<PrevPressT> > >
        ( Plato::ScalarMultiVectorT<PrevPressT>("previous pressure", tNumCells, PhysicsT::mNumMassDofsPerCell) );
    auto tHostPrevPress = Kokkos::create_mirror(tPrevPress->mData);
    tHostPrevPress(0, 0) = 1; tHostPrevPress(1, 0) = 4;
    tHostPrevPress(0, 1) = 2; tHostPrevPress(1, 1) = 5;
    tHostPrevPress(0, 2) = 3; tHostPrevPress(1, 2) = 6;
    Kokkos::deep_copy(tPrevPress->mData, tHostPrevPress);
    tWorkSets.set("previous pressure", tPrevPress);

    using CurPressT = EvaluationT::CurrentMassScalarType;
    auto tCurPress = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<CurPressT> > >
        ( Plato::ScalarMultiVectorT<CurPressT>("current pressure", tNumCells, PhysicsT::mNumMassDofsPerCell) );
    auto tHostCurPress = Kokkos::create_mirror(tCurPress->mData);
    tHostCurPress(0, 0) = 7; tHostCurPress(1, 0) = 10;
    tHostCurPress(0, 1) = 8; tHostCurPress(1, 1) = 11;
    tHostCurPress(0, 2) = 9; tHostCurPress(1, 2) = 12;
    Kokkos::deep_copy(tCurPress->mData, tHostCurPress);
    tWorkSets.set("current pressure", tCurPress);

    auto tTimeStep = std::make_shared< Plato::MetaData< Plato::ScalarVector > >( Plato::ScalarVector("time step", 1) );
    auto tHostTimeStep = Kokkos::create_mirror(tTimeStep->mData);
    tHostTimeStep(0) = 0.01;
    Kokkos::deep_copy(tTimeStep->mData, tHostTimeStep);
    tWorkSets.set("critical time step", tTimeStep);

    TEST_EQUALITY(3,tNumNodesPerCell);
    auto tAC = std::make_shared< Plato::MetaData< Plato::ScalarMultiVector > >
        ( Plato::ScalarMultiVector("artificial compressibility", tNumCells, tNumNodesPerCell) );
    auto tHostAC = Kokkos::create_mirror(tAC->mData);
    tHostAC(0, 0) = 0.1; tHostAC(1, 0) = 0.4;
    tHostAC(0, 1) = 0.2; tHostAC(1, 1) = 0.5;
    tHostAC(0, 2) = 0.3; tHostAC(1, 2) = 0.6;
    Kokkos::deep_copy(tAC->mData, tHostAC);
    tWorkSets.set("artificial compressibility", tAC);

    // evaluate pressure increment residual
    Plato::DataMap tDataMap;
    Plato::ScalarMultiVectorT<EvaluationT::ResultScalarType> tResult("result", tNumCells, PhysicsT::mNumMassDofsPerCell);
    Plato::Fluids::PressureResidual<PhysicsT,EvaluationT> tResidual(tDomain,tDataMap);
    tResidual.evaluate(tWorkSets, tResult);

    // test values
    // TODO: FIX DUE TO CRITICAL TIME STEP CHANGES
    auto tTol = 1e-4;
    auto tHostResult = Kokkos::create_mirror(tResult);
    Kokkos::deep_copy(tHostResult, tResult);
    std::vector<std::vector<Plato::Scalar>> tGold =
        {{10.008225,5.0055,3.30055833333333},{2.4006,1.98625,1.832566666666667}};
    for (Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++)
    {
        for (Plato::OrdinalType tDof = 0; tDof < tNumNodesPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(tGold[tCell][tDof], tHostResult(tCell, tDof), tTol);
        }
    }
    //Plato::print_array_2D(tResult, "results");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PressureIncrementResidual_ThetaTwo_Set)
{
    // set physics and evaluation type
    constexpr Plato::OrdinalType tNumSpaceDims = 2;
    using PhysicsT = Plato::IncompressibleFluids<tNumSpaceDims>::MassPhysicsT;
    using EvaluationT = Plato::Fluids::Evaluation<PhysicsT::SimplexT>::Residual;

    // build mesh, spatial domain, and spatial model
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);
    auto tMeshSets = PlatoUtestHelpers::get_box_mesh_sets(tMesh.operator*());
    auto tSpatialDomainName = std::string("my box");
    Plato::SpatialDomain tDomain(tMesh.operator*(), tMeshSets, tSpatialDomainName);
    auto tElementBlockName = std::string("body");
    tDomain.cellOrdinals(tElementBlockName);

    // set workset
    Plato::WorkSets tWorkSets;
    auto tNumCells = tMesh->nelems();
    constexpr auto tNumNodesPerCell = tNumSpaceDims + 1;
    using ConfigT = EvaluationT::ConfigScalarType;
    Plato::NodeCoordinate<tNumSpaceDims> tNodeCoordinate( (&tMesh.operator*()) );
    auto tConfig = std::make_shared< Plato::MetaData< Plato::ScalarArray3DT<ConfigT> > >
        ( Plato::ScalarArray3DT<ConfigT>("configuration", tNumCells, tNumNodesPerCell, tNumSpaceDims) );
    Plato::workset_config_scalar<tNumSpaceDims, tNumNodesPerCell>(tMesh->nelems(), tNodeCoordinate, tConfig->mData);
    tWorkSets.set("configuration", tConfig);

    using PrevVelT = EvaluationT::PreviousMomentumScalarType;
    auto tPrevVel = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<PrevVelT> > >
        ( Plato::ScalarMultiVectorT<PrevVelT>("previous velocity", tNumCells, PhysicsT::mNumMomentumDofsPerCell) );
    auto tHostVelocity = Kokkos::create_mirror(tPrevVel->mData);
    tHostVelocity(0, 0) = 1; tHostVelocity(1, 0) = 7;
    tHostVelocity(0, 1) = 2; tHostVelocity(1, 1) = 8;
    tHostVelocity(0, 2) = 3; tHostVelocity(1, 2) = 9;
    tHostVelocity(0, 3) = 4; tHostVelocity(1, 3) = 10;
    tHostVelocity(0, 4) = 5; tHostVelocity(1, 4) = 11;
    tHostVelocity(0, 5) = 6; tHostVelocity(1, 5) = 12;
    Kokkos::deep_copy(tPrevVel->mData, tHostVelocity);
    tWorkSets.set("previous velocity", tPrevVel);

    using PredictorT = EvaluationT::MomentumPredictorScalarType;
    auto tPredictor = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<PredictorT> > >
        ( Plato::ScalarMultiVectorT<PredictorT>("predictor", tNumCells, PhysicsT::mNumMomentumDofsPerCell) );
    auto tHostPredictor = Kokkos::create_mirror(tPredictor->mData);
    tHostPredictor(0, 0) = 0.1; tHostPredictor(1, 0) = 0.7;
    tHostPredictor(0, 1) = 0.2; tHostPredictor(1, 1) = 0.8;
    tHostPredictor(0, 2) = 0.3; tHostPredictor(1, 2) = 0.9;
    tHostPredictor(0, 3) = 0.4; tHostPredictor(1, 3) = 1.0;
    tHostPredictor(0, 4) = 0.5; tHostPredictor(1, 4) = 1.1;
    tHostPredictor(0, 5) = 0.6; tHostPredictor(1, 5) = 1.2;
    Kokkos::deep_copy(tPredictor->mData, tHostPredictor);
    tWorkSets.set("current predictor", tPredictor);

    using PrevPressT = EvaluationT::PreviousMassScalarType;
    auto tPrevPress = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<PrevPressT> > >
        ( Plato::ScalarMultiVectorT<PrevPressT>("previous pressure", tNumCells, PhysicsT::mNumMassDofsPerCell) );
    auto tHostPrevPress = Kokkos::create_mirror(tPrevPress->mData);
    tHostPrevPress(0, 0) = 1; tHostPrevPress(1, 0) = 4;
    tHostPrevPress(0, 1) = 2; tHostPrevPress(1, 1) = 5;
    tHostPrevPress(0, 2) = 3; tHostPrevPress(1, 2) = 6;
    Kokkos::deep_copy(tPrevPress->mData, tHostPrevPress);
    tWorkSets.set("previous pressure", tPrevPress);

    using CurPressT = EvaluationT::CurrentMassScalarType;
    auto tCurPress = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<CurPressT> > >
        ( Plato::ScalarMultiVectorT<CurPressT>("current pressure", tNumCells, PhysicsT::mNumMassDofsPerCell) );
    auto tHostCurPress = Kokkos::create_mirror(tCurPress->mData);
    tHostCurPress(0, 0) = 1; tHostCurPress(1, 0) = 3;
    tHostCurPress(0, 1) = 8; tHostCurPress(1, 1) = 11;
    tHostCurPress(0, 2) = 2; tHostCurPress(1, 2) = 4;
    Kokkos::deep_copy(tCurPress->mData, tHostCurPress);
    tWorkSets.set("current pressure", tCurPress);

    auto tTimeStep = std::make_shared< Plato::MetaData< Plato::ScalarVector > >( Plato::ScalarVector("time step", 1) );
    auto tHostTimeStep = Kokkos::create_mirror(tTimeStep->mData);
    tHostTimeStep(0) = 0.01;
    Kokkos::deep_copy(tTimeStep->mData, tHostTimeStep);
    tWorkSets.set("critical time step", tTimeStep);

    TEST_EQUALITY(3,tNumNodesPerCell);
    auto tAC = std::make_shared< Plato::MetaData< Plato::ScalarMultiVector > >
        ( Plato::ScalarMultiVector("artificial compressibility", tNumCells, tNumNodesPerCell) );
    auto tHostAC = Kokkos::create_mirror(tAC->mData);
    tHostAC(0, 0) = 0.1; tHostAC(1, 0) = 0.4;
    tHostAC(0, 1) = 0.2; tHostAC(1, 1) = 0.5;
    tHostAC(0, 2) = 0.3; tHostAC(1, 2) = 0.6;
    Kokkos::deep_copy(tAC->mData, tHostAC);
    tWorkSets.set("artificial compressibility", tAC);

    // evaluate pressure increment residual
    Plato::DataMap tDataMap;
    Plato::ScalarMultiVectorT<EvaluationT::ResultScalarType> tResult("result", tNumCells, PhysicsT::mNumMassDofsPerCell);
    Plato::Fluids::PressureResidual<PhysicsT,EvaluationT> tResidual(tDomain,tDataMap);
    tResidual.evaluate(tWorkSets, tResult);

    // test values
    // TODO: FIX: IT WILL FAIL DUE TO CHANGES ON HOW THE CIRITCAL TIME STEP IS CALCULATED
    auto tTol = 1e-4;
    auto tHostResult = Kokkos::create_mirror(tResult);
    Kokkos::deep_copy(tHostResult, tResult);
    std::vector<std::vector<Plato::Scalar>> tGold =
        {{2.7858527,1.3956888,0.8915759},{0.31446667,0.3289583,0.43647778}};
    for (Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++)
    {
        for (Plato::OrdinalType tDof = 0; tDof < tNumNodesPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(tGold[tCell][tDof], tHostResult(tCell, tDof), tTol);
        }
    }
    //Plato::print_array_2D(tResult, "results");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, IntegrateInertialForces)
{
    constexpr Plato::OrdinalType tNumCells = 2;
    constexpr Plato::OrdinalType tSpaceDims = 2;
    constexpr auto tNumNodesPerCell = tSpaceDims + 1;
    Plato::ScalarVector tBasisFunctions("basis functions", tNumNodesPerCell);
    Plato::blas1::fill(0.33333333333333333333333, tBasisFunctions);
    Plato::ScalarVector tCellVolume("cell weight", tNumCells);
    Plato::blas1::fill(0.5, tCellVolume);
    Plato::ScalarVector tCurPress("current pressure", tNumCells);
    auto tHostCurPress = Kokkos::create_mirror(tCurPress);
    tHostCurPress(0) = 7; tHostCurPress(1) = 8;
    Kokkos::deep_copy(tCurPress, tHostCurPress);
    Plato::ScalarVector tPrevPress("previous pressure", tNumCells);
    auto tHostPrevPress = Kokkos::create_mirror(tPrevPress);
    tHostPrevPress(0) = 1; tHostPrevPress(1) = 2;
    Kokkos::deep_copy(tPrevPress, tHostPrevPress);
    Plato::ScalarMultiVector tArtificialCompress("artificial compressibility", tNumCells, tNumNodesPerCell);
    auto tHostArtificialCompress = Kokkos::create_mirror(tArtificialCompress);
    tHostArtificialCompress(0,0) = 0.1; tHostArtificialCompress(0,1) = 0.2; tHostArtificialCompress(0,2) = 0.3;
    tHostArtificialCompress(1,0) = 0.4; tHostArtificialCompress(1,1) = 0.5; tHostArtificialCompress(1,2) = 0.6;
    Kokkos::deep_copy(tArtificialCompress, tHostArtificialCompress);
    Plato::ScalarMultiVector tResult("result", tNumCells, tNumNodesPerCell);

    // call device function
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        Plato::Fluids::integrate_inertial_pressure_forces<tNumNodesPerCell>
            (aCellOrdinal, tBasisFunctions, tCellVolume, tCurPress, tPrevPress, tArtificialCompress, tResult);
    }, "unit test integrate_inertial_pressure_forces");

    // test values
    auto tTol = 1e-4;
    auto tHostResult = Kokkos::create_mirror(tResult);
    Kokkos::deep_copy(tHostResult, tResult);
    std::vector<std::vector<Plato::Scalar>> tGold = {{10.0,5.0,3.333333},{2.5,2.0,1.666667}};
    for (Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++)
    {
        for (Plato::OrdinalType tDof = 0; tDof < tNumNodesPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(tGold[tCell][tDof], tHostResult(tCell, tDof), tTol);
        }
    }
    //Plato::print_array_2D(tResult, "results");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, DeltaPressureGradientDivergence)
{
    constexpr Plato::OrdinalType tNumCells = 2;
    constexpr Plato::OrdinalType tSpaceDims = 2;
    constexpr auto tNumNodesPerCell = tSpaceDims + 1;
    Plato::ScalarMultiVector tResult("result", tNumCells, tNumNodesPerCell);
    Plato::ScalarMultiVector tTimeStep("time step", tNumCells, tNumNodesPerCell);
    Plato::blas2::fill(0.1, tTimeStep);
    Plato::ScalarMultiVector tPrevPressGrad("previous pressure gradient", tNumCells, tSpaceDims);
    auto tHostPrevPressGrad = Kokkos::create_mirror(tPrevPressGrad);
    tHostPrevPressGrad(0,0) = 1; tHostPrevPressGrad(0,1) = 2;
    tHostPrevPressGrad(1,0) = 3; tHostPrevPressGrad(1,1) = 4;
    Kokkos::deep_copy(tPrevPressGrad, tHostPrevPressGrad);
    Plato::ScalarMultiVector tCurPressGrad("current pressure gradient", tNumCells, tSpaceDims);
    auto tHostCurPressGrad = Kokkos::create_mirror(tCurPressGrad);
    tHostCurPressGrad(0,0) = 5; tHostCurPressGrad(0,1) = 6;
    tHostCurPressGrad(1,0) = 7; tHostCurPressGrad(1,1) = 8;
    Kokkos::deep_copy(tCurPressGrad, tHostCurPressGrad);
    Plato::ScalarArray3D tGradient("cell gradient", tNumCells, tNumNodesPerCell, tSpaceDims);
    auto tHostGradient = Kokkos::create_mirror(tGradient);
    tHostGradient(0,0,0) = -1; tHostGradient(0,0,1) = 0;
    tHostGradient(0,1,0) = 1;  tHostGradient(0,1,1) = -1;
    tHostGradient(0,2,0) = 0;  tHostGradient(0,2,1) = 1;
    tHostGradient(1,0,0) = 1;  tHostGradient(1,0,1) = 0;
    tHostGradient(1,1,0) = -1; tHostGradient(1,1,1) = 1;
    tHostGradient(1,2,0) = 0;  tHostGradient(1,2,1) = -1;
    Kokkos::deep_copy(tGradient, tHostGradient);
    Plato::ScalarVector tCellVolume("cell weight", tNumCells);
    Plato::blas1::fill(0.5, tCellVolume);

    // call device function
    auto tMultiplier = 1.0;
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        Plato::Fluids::integrate_divergence_delta_pressure_gradient<tNumNodesPerCell,tSpaceDims>
            (aCellOrdinal, tTimeStep, tGradient, tCellVolume, tCurPressGrad, tPrevPressGrad, tResult, tMultiplier);
    }, "unit test integrate_divergence_delta_pressure_gradient");

    // test values
    auto tTol = 1e-4;
    auto tHostResult = Kokkos::create_mirror(tResult);
    Kokkos::deep_copy(tHostResult, tResult);
    std::vector<std::vector<Plato::Scalar>> tGold = {{-0.2,0.0,0.2},{0.2,0.0,-0.2}};
    for (Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++)
    {
        for (Plato::OrdinalType tDof = 0; tDof < tNumNodesPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(tGold[tCell][tDof], tHostResult(tCell, tDof), tTol);
        }
    }
    //Plato::print_array_2D(tResult, "results");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, CalculateScalarFieldGradient)
{
    constexpr Plato::OrdinalType tNumCells = 2;
    constexpr Plato::OrdinalType tSpaceDims = 2;
    constexpr auto tNumNodesPerCell = tSpaceDims + 1;
    Plato::ScalarMultiVector tResult("result", tNumCells, tSpaceDims);
    Plato::ScalarArray3D tGradient("gradient", tNumCells, tNumNodesPerCell, tSpaceDims);
    auto tHostGradient = Kokkos::create_mirror(tGradient);
    tHostGradient(0,0,0) = -1; tHostGradient(0,0,1) = 0;
    tHostGradient(0,1,0) = 1;  tHostGradient(0,1,1) = -1;
    tHostGradient(0,2,0) = 0;  tHostGradient(0,2,1) = 1;
    tHostGradient(1,0,0) = 1;  tHostGradient(1,0,1) = 0;
    tHostGradient(1,1,0) = -1; tHostGradient(1,1,1) = 1;
    tHostGradient(1,2,0) = 0;  tHostGradient(1,2,1) = -1;
    Kokkos::deep_copy(tGradient, tHostGradient);
    Plato::ScalarMultiVector tPressure("pressure", tNumCells, tNumNodesPerCell);
    auto tHostPressure = Kokkos::create_mirror(tPressure);
    tHostPressure(0,0) = 1; tHostPressure(0,1) = 2; tHostPressure(0,2) = 3;
    tHostPressure(1,0) = 4; tHostPressure(1,1) = 5; tHostPressure(1,2) = 6;
    Kokkos::deep_copy(tPressure, tHostPressure);

    // call device function
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        Plato::Fluids::calculate_scalar_field_gradient<tNumNodesPerCell,tSpaceDims>(aCellOrdinal, tGradient, tPressure, tResult);
    }, "unit test calculate_scalar_field_gradient");

    // test values
    auto tTol = 1e-4;
    auto tHostResult = Kokkos::create_mirror(tResult);
    Kokkos::deep_copy(tHostResult, tResult);
    std::vector<std::vector<Plato::Scalar>> tGold = {{1.0,1.0},{-1.0,-1.0}};
    for (Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++)
    {
        for (Plato::OrdinalType tDof = 0; tDof < tSpaceDims; tDof++)
        {
            TEST_FLOATING_EQUALITY(tGold[tCell][tDof], tHostResult(tCell, tDof), tTol);
        }
    }
    //Plato::print_array_2D(tResult, "results");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PreviousPressureGradientDivergence)
{
    constexpr Plato::OrdinalType tNumCells = 2;
    constexpr Plato::OrdinalType tSpaceDims = 2;
    constexpr auto tNumNodesPerCell = tSpaceDims + 1;
    Plato::ScalarMultiVector tResult("result", tNumCells, tNumNodesPerCell);
    Plato::ScalarMultiVector tTimeStep("time step", tNumCells, tNumNodesPerCell);
    Plato::blas2::fill(0.1, tTimeStep);
    Plato::ScalarMultiVector tPressureGrad("pressure gradient", tNumCells, tSpaceDims);
    auto tHostPressureGrad = Kokkos::create_mirror(tPressureGrad);
    tHostPressureGrad(0,0) = 1; tHostPressureGrad(0,1) = 2;
    tHostPressureGrad(1,0) = 3; tHostPressureGrad(1,1) = 4;
    Kokkos::deep_copy(tPressureGrad, tHostPressureGrad);
    Plato::ScalarArray3D tGradient("cell gradient", tNumCells, tNumNodesPerCell, tSpaceDims);
    auto tHostGradient = Kokkos::create_mirror(tGradient);
    tHostGradient(0,0,0) = -1; tHostGradient(0,0,1) = 0;
    tHostGradient(0,1,0) = 1;  tHostGradient(0,1,1) = -1;
    tHostGradient(0,2,0) = 0;  tHostGradient(0,2,1) = 1;
    tHostGradient(1,0,0) = 1;  tHostGradient(1,0,1) = 0;
    tHostGradient(1,1,0) = -1; tHostGradient(1,1,1) = 1;
    tHostGradient(1,2,0) = 0;  tHostGradient(1,2,1) = -1;
    Kokkos::deep_copy(tGradient, tHostGradient);
    Plato::ScalarVector tCellVolume("cell weight", tNumCells);
    Plato::blas1::fill(0.5, tCellVolume);

    // call device function
    auto tMultiplier = 1.0;
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        Plato::Fluids::integrate_divergence_previous_pressure_gradient<tNumNodesPerCell,tSpaceDims>
            (aCellOrdinal, tTimeStep, tGradient, tCellVolume, tPressureGrad, tResult, tMultiplier);
    }, "unit test integrate_divergence_previous_pressure_gradient");

    // test values
    auto tTol = 1e-4;
    auto tHostResult = Kokkos::create_mirror(tResult);
    Kokkos::deep_copy(tHostResult, tResult);
    std::vector<std::vector<Plato::Scalar>> tGold = {{-0.05,-0.05,0.1},{0.15,0.05,-0.2}};
    for (Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++)
    {
        for (Plato::OrdinalType tDof = 0; tDof < tNumNodesPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(tGold[tCell][tDof], tHostResult(tCell, tDof), tTol);
        }
    }
    //Plato::print_array_2D(tResult, "results");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, IntegrateAdvectedForces)
{
    constexpr Plato::OrdinalType tNumCells = 2;
    constexpr Plato::OrdinalType tSpaceDims = 2;
    constexpr auto tNumNodesPerCell = tSpaceDims + 1;
    Plato::ScalarMultiVector tResult("result", tNumCells, tNumNodesPerCell);
    Plato::ScalarMultiVector tPrevVel("previous velocity", tNumCells, tSpaceDims);
    auto tHostPrevVel = Kokkos::create_mirror(tPrevVel);
    tHostPrevVel(0,0) = 1; tHostPrevVel(0,1) = 2;
    tHostPrevVel(1,0) = 3; tHostPrevVel(1,1) = 4;
    Kokkos::deep_copy(tPrevVel, tHostPrevVel);
    Plato::ScalarArray3D tGradient("cell gradient", tNumCells, tNumNodesPerCell, tSpaceDims);
    auto tHostGradient = Kokkos::create_mirror(tGradient);
    tHostGradient(0,0,0) = -1; tHostGradient(0,0,1) = 0;
    tHostGradient(0,1,0) = 1;  tHostGradient(0,1,1) = -1;
    tHostGradient(0,2,0) = 0;  tHostGradient(0,2,1) = 1;
    tHostGradient(1,0,0) = 1;  tHostGradient(1,0,1) = 0;
    tHostGradient(1,1,0) = -1; tHostGradient(1,1,1) = 1;
    tHostGradient(1,2,0) = 0;  tHostGradient(1,2,1) = -1;
    Kokkos::deep_copy(tGradient, tHostGradient);
    Plato::ScalarVector tCellVolume("cell weight", tNumCells);
    Plato::blas1::fill(0.5, tCellVolume);
    Plato::ScalarVector tBasisFunctions("basis functions", tNumNodesPerCell);
    Plato::blas1::fill(0.33333333333333333333333, tBasisFunctions);

    // call device function
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        Plato::Fluids::integrate_divergence_operator<tNumNodesPerCell,tSpaceDims>
            (aCellOrdinal, tBasisFunctions, tGradient, tCellVolume, tPrevVel, tResult);
    }, "unit test integrate_divergence_operator");

    // test values
    auto tTol = 1e-4;
    auto tHostResult = Kokkos::create_mirror(tResult);
    Kokkos::deep_copy(tHostResult, tResult);
    std::vector<std::vector<Plato::Scalar>> tGold = {{-0.5,-0.5,1.0},{1.5,0.5,-2.0}};
    for (Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++)
    {
        for (Plato::OrdinalType tDof = 0; tDof < tNumNodesPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(tGold[tCell][tDof], tHostResult(tCell, tDof), tTol);
        }
    }
    //Plato::print_array_2D(tResult, "results");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, IntegrateDeltaAdvectedForces)
{
    constexpr Plato::OrdinalType tNumCells = 2;
    constexpr Plato::OrdinalType tSpaceDims = 2;
    constexpr auto tNumNodesPerCell = tSpaceDims + 1;
    Plato::ScalarMultiVector tResult("result", tNumCells, tNumNodesPerCell);
    Plato::ScalarMultiVector tPrevVel("previous velocity", tNumCells, tSpaceDims);
    auto tHostPrevVel = Kokkos::create_mirror(tPrevVel);
    tHostPrevVel(0,0) = 1; tHostPrevVel(0,1) = 2;
    tHostPrevVel(1,0) = 3; tHostPrevVel(1,1) = 4;
    Kokkos::deep_copy(tPrevVel, tHostPrevVel);
    Plato::ScalarMultiVector tPredVel("predicted velocity", tNumCells, tSpaceDims);
    auto tHostPredVel = Kokkos::create_mirror(tPredVel);
    tHostPredVel(0,0) = 11; tHostPredVel(0,1) = 12;
    tHostPredVel(1,0) = 13; tHostPredVel(1,1) = 14;
    Kokkos::deep_copy(tPredVel, tHostPredVel);
    Plato::ScalarArray3D tGradient("cell gradient", tNumCells, tNumNodesPerCell, tSpaceDims);
    auto tHostGradient = Kokkos::create_mirror(tGradient);
    tHostGradient(0,0,0) = -1; tHostGradient(0,0,1) = 0;
    tHostGradient(0,1,0) = 1;  tHostGradient(0,1,1) = -1;
    tHostGradient(0,2,0) = 0;  tHostGradient(0,2,1) = 1;
    tHostGradient(1,0,0) = 1;  tHostGradient(1,0,1) = 0;
    tHostGradient(1,1,0) = -1; tHostGradient(1,1,1) = 1;
    tHostGradient(1,2,0) = 0;  tHostGradient(1,2,1) = -1;
    Kokkos::deep_copy(tGradient, tHostGradient);
    Plato::ScalarVector tCellVolume("cell weight", tNumCells);
    Plato::blas1::fill(0.5, tCellVolume);

    // call device function
    auto tMultiplier = 1.0;
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        Plato::Fluids::integrate_divergence_delta_predicted_momentum<tNumNodesPerCell,tSpaceDims>
            (aCellOrdinal, tGradient, tCellVolume, tPredVel, tPrevVel, tResult, tMultiplier);
    }, "unit test integrate_divergence_delta_predicted_momentum");

    // test values
    auto tTol = 1e-4;
    auto tHostResult = Kokkos::create_mirror(tResult);
    Kokkos::deep_copy(tHostResult, tResult);
    std::vector<std::vector<Plato::Scalar>> tGold = {{-5.0,0.0,5.0},{5.0,0.0,-5.0}};
    for (Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++)
    {
        for (Plato::OrdinalType tDof = 0; tDof < tNumNodesPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(tGold[tCell][tDof], tHostResult(tCell, tDof), tTol);
        }
    }
    //Plato::print_array_2D(tResult, "results");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, SIMP_TemperatureResidual)
{
    // set xml file inputs
    Teuchos::RCP<Teuchos::ParameterList> tInputs =
        Teuchos::getParametersFromXmlString(
            "<ParameterList name='Plato Problem'>"
            "  <ParameterList name='Hyperbolic'>"
            "    <Parameter name='Scenario' type='string' value='Density TO'/>"
            "    <Parameter name='Heat Transfer' type='string' value='Natural'/>"
            "    <ParameterList  name='Dimensionless Properties'>"
            "      <Parameter  name='Characteristic Length'  type='double'  value='1.0'/>"
            "    </ParameterList>"
            "    <ParameterList name='Energy Conservation'>"
            "      <ParameterList name='Penalty Function'>"
            "        <Parameter  name='Heat Source Penalty Exponent'  type='double' value='3.0'/>"
            "        <Parameter  name='Thermal Diffusion Penalty Exponent'  type='double' value='3.0'/>"
            "      </ParameterList>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList name='Spatial Model'>"
            "    <ParameterList name='Domains'>"
            "      <ParameterList name='Design Volume'>"
            "        <Parameter name='Element Block' type='string' value='block_1'/>"
            "        <Parameter name='Material Model' type='string' value='Steel'/>"
            "      </ParameterList>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList name='Material Models'>"
            "    <ParameterList name='Steel'>"
            "      <ParameterList name='Thermal Properties'>"
            "        <Parameter  name='Thermal Conductivity'  type='double'  value='1'/>"
            "        <Parameter  name='Reference Temperature'       type='double'  value='10.0'/>"
            "        <Parameter  name='Thermal Diffusivity Ratio'   type='double'  value='1.0'/>"
            "      </ParameterList>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList  name='Heat Source'>"
            "    <Parameter  name='Constant'  type='double'  value='2.0'/>"
            "  </ParameterList>"
            "</ParameterList>"
            );

    // build mesh, spatial domain, and spatial model
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);
    auto tMeshSets = PlatoUtestHelpers::get_box_mesh_sets(tMesh.operator*());
    Plato::SpatialDomain tDomain(tMesh.operator*(), tMeshSets, "box");
    tDomain.cellOrdinals("body");
    tDomain.setMaterialName("Steel");
    tDomain.setElementBlockName("block_1");
    Plato::SpatialModel tModel(tMesh.operator*(), tMeshSets);
    tModel.append(tDomain);

    // set physics and evaluation type
    constexpr Plato::OrdinalType tNumSpaceDims = 2;
    using PhysicsT = Plato::IncompressibleFluids<tNumSpaceDims>::EnergyPhysicsT;

    // set control variables
    auto tNumNodes = tMesh->nverts();
    Plato::ScalarVector tControls("control", tNumNodes);
    Plato::blas1::fill(1.0, tControls);

    // set state variables
    Plato::Primal tVariables;
    auto tNumVelDofs = tNumNodes * tNumSpaceDims;
    Plato::ScalarVector tPrevVel("previous velocity", tNumVelDofs);
    auto tHostPrevVel = Kokkos::create_mirror(tPrevVel);
    tHostPrevVel(0) = 1; tHostPrevVel(1) = 2; tHostPrevVel(2) = 3; tHostPrevVel(3) = 4;
    tHostPrevVel(4) = 5; tHostPrevVel(5) = 6; tHostPrevVel(6) = 7; tHostPrevVel(7) = 8;
    Kokkos::deep_copy(tPrevVel, tHostPrevVel);
    tVariables.vector("previous velocity", tPrevVel);
    Plato::ScalarVector tPrevPress("previous pressure", tNumNodes);
    Plato::blas1::fill(1.0, tPrevPress);
    tVariables.vector("previous pressure", tPrevPress);
    Plato::ScalarVector tPrevTemp("previous temperature", tNumNodes);
    auto tHostPrevTemp = Kokkos::create_mirror(tPrevTemp);
    tHostPrevTemp(0) = 1; tHostPrevTemp(1) = 2; tHostPrevTemp(2) = 3; tHostPrevTemp(3) = 4;
    Kokkos::deep_copy(tPrevTemp, tHostPrevTemp);
    tVariables.vector("previous temperature", tPrevTemp);

    Plato::ScalarVector tCurVel("current velocity", tNumVelDofs);
    tVariables.vector("current velocity", tCurVel);
    Plato::ScalarVector tCurPred("current predictor", tNumVelDofs);
    tVariables.vector("current predictor", tCurPred);
    Plato::ScalarVector tCurPress("current pressure", tNumNodes);
    tVariables.vector("current pressure", tCurPress);
    Plato::ScalarVector tCurTemp("current temperature", tNumNodes);
    tVariables.vector("current temperature", tCurTemp);

    Plato::ScalarVector tTimeSteps("time step", 1);
    Plato::blas1::fill(0.1, tTimeSteps);
    tVariables.vector("critical time step", tTimeSteps);

    // allocate vector function
    Plato::DataMap tDataMap;
    std::string tFuncName("Temperature");
    Plato::Fluids::VectorFunction<PhysicsT> tVectorFunction(tFuncName, tModel, tDataMap, tInputs.operator*());

    // test vector function value
    auto tResidual = tVectorFunction.value(tControls, tVariables);

    auto tTol = 1e-4;
    auto tHostResidual = Kokkos::create_mirror(tResidual);
    Kokkos::deep_copy(tHostResidual, tResidual);
    std::vector<Plato::Scalar> tGold = {-0.615278,-0.0647222,-0.1075,0.000833333};
    for(auto& tValue : tGold)
    {
        auto tIndex = &tValue - &tGold[0];
        TEST_FLOATING_EQUALITY(tValue,tHostResidual(tIndex),tTol);
    }
    //Plato::print(tResidual, "residual");
}

/*
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, IntegrateStabilizingScalarForces)
{
    // build mesh, mesh sets, and spatial domain
    constexpr auto tSpaceDims = 2;
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);

    // set input data for unit test
    auto tNumCells = tMesh->nelems();
    constexpr auto tNumNodesPerCell = tSpaceDims + 1;
    Plato::ScalarVector tStabForce("cell weight", tNumCells);
    auto tHostStabForce = Kokkos::create_mirror(tStabForce);
    tHostStabForce(0) = 1; tHostStabForce(1) = -1;
    Kokkos::deep_copy(tStabForce, tHostStabForce);
    Plato::ScalarVector tDivergence("cell weight", tNumCells);
    auto tHostDivergence = Kokkos::create_mirror(tDivergence);
    tHostDivergence(0) = 4; tHostDivergence(1) = -4;
    Kokkos::deep_copy(tDivergence, tHostDivergence);
    Plato::ScalarVector tCellVolume("cell weight", tNumCells);
    Plato::blas1::fill(0.5, tCellVolume);
    Plato::ScalarVector tBasisFunctions("basis functions", tNumNodesPerCell);
    Plato::blas1::fill(0.33333333333333333333333, tBasisFunctions);
    Plato::ScalarArray3D tGradient("cell gradient", tNumCells, tNumNodesPerCell, tSpaceDims);
    auto tHostGradient = Kokkos::create_mirror(tGradient);
    tHostGradient(0,0,0) = -1; tHostGradient(0,0,1) = 0;
    tHostGradient(0,1,0) = 1;  tHostGradient(0,1,1) = -1;
    tHostGradient(0,2,0) = 0;  tHostGradient(0,2,1) = 1;
    tHostGradient(1,0,0) = 1;  tHostGradient(1,0,1) = 0;
    tHostGradient(1,1,0) = -1; tHostGradient(1,1,1) = 1;
    tHostGradient(1,2,0) = 0;  tHostGradient(1,2,1) = -1;
    Kokkos::deep_copy(tGradient, tHostGradient);
    Plato::ScalarMultiVector tPrevVelGP("previous velocities", tNumCells, tSpaceDims);
    auto tHostPrevVelGP = Kokkos::create_mirror(tPrevVelGP);
    tHostPrevVelGP(0,0) = 1; tHostPrevVelGP(0,1) = 2;
    tHostPrevVelGP(1,0) = 3; tHostPrevVelGP(1,1) = 4;
    Kokkos::deep_copy(tPrevVelGP, tHostPrevVelGP);
    Plato::ScalarMultiVector tResult("result", tNumCells, tNumNodesPerCell);

    // call device function
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        Plato::Fluids::integrate_stabilizing_scalar_forces<tNumNodesPerCell,tSpaceDims>
            (aCellOrdinal, tCellVolume, tGradient, tPrevVelGP, tStabForce, tResult);
    }, "unit test integrate_stabilizing_scalar_forces");

    auto tTol = 1e-4;
    std::vector<std::vector<Plato::Scalar>> tGold =
        {{0.166666666666667,0.166666666666667,1.666666666666667},
         {-0.833333333333333,0.166666666666667,2.666666666666667}};
    auto tHostResult = Kokkos::create_mirror(tResult);
    Kokkos::deep_copy(tHostResult, tResult);
    for(auto& tGArray : tGold)
    {
        auto tCell = &tGArray - &tGold[0];
        for(auto& tGValue : tGArray)
        {
            auto tDof = &tGValue - &tGArray[0];
            TEST_FLOATING_EQUALITY(tGValue,tHostResult(tCell,tDof),tTol);
        }
    }
    //Plato::print_array_2D(tResult, "result");
}
*/

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, CalculateInertialForces_ThermalResidual)
{
    // set input data for unit test
    constexpr auto tNumCells = 2;
    constexpr auto tSpaceDims = 2;
    constexpr auto tNumNodesPerCell = tSpaceDims + 1;
    Plato::ScalarVector tVolume("result", tNumCells);
    Plato::blas1::fill(1.0, tVolume);
    Plato::ScalarVector tCurTemp("current temperature", tNumCells);
    auto tHostCurTemp = Kokkos::create_mirror(tCurTemp);
    tHostCurTemp(0,0) = 10; tHostCurTemp(0,1) = 10; tHostCurTemp(0,2) = 10;
    tHostCurTemp(1,0) = 12; tHostCurTemp(1,1) = 12; tHostCurTemp(1,2) = 12;
    Kokkos::deep_copy(tCurTemp, tHostCurTemp);
    Plato::ScalarVector tPrevTemp("previous temperature", tNumCells);
    auto tHostPrevTemp = Kokkos::create_mirror(tPrevTemp);
    tHostPrevTemp(0,0) = 5; tHostPrevTemp(0,1) = 5; tHostPrevTemp(0,2) = 5;
    tHostPrevTemp(1,0) = 6; tHostPrevTemp(1,1) = 6; tHostPrevTemp(1,2) = 6;
    Kokkos::deep_copy(tPrevTemp, tHostPrevTemp);
    Plato::ScalarMultiVector tResult("flux", tNumCells, tNumNodesPerCell);

    // call device function
    Plato::LinearTetCubRuleDegreeOne<tSpaceDims> tCubRule;
    auto tBasisFunctions = tCubRule.getBasisFunctions();
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        Plato::Fluids::calculate_inertial_forces<tNumNodesPerCell>(aCellOrdinal, tBasisFunctions, tVolume, tCurTemp, tPrevTemp, tResult);
    }, "unit test calculate_inertial_forces");

    auto tTol = 1e-4;
    std::vector<std::vector<Plato::Scalar>> tGold =
        {{1.666666666666667,1.666666666666667,1.666666666666667},
         {2.0,2.0,2.0}};
    auto tHostResult = Kokkos::create_mirror(tResult);
    Kokkos::deep_copy(tHostResult, tResult);
    for(auto& tGArray : tGold)
    {
        auto tCell = &tGArray - &tGold[0];
        for(auto& tGValue : tGArray)
        {
            auto tDof = &tGValue - &tGArray[0];
            TEST_FLOATING_EQUALITY(tGValue,tHostResult(tCell,tDof),tTol);
        }
    }
    //Plato::print_array_2D(tFlux, "flux");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PenalizeHeatSourceConstant)
{
    // set input data for unit test
    constexpr auto tNumCells = 2;
    constexpr auto tSpaceDims = 2;
    constexpr auto tNumNodesPerCell = tSpaceDims + 1;

    constexpr auto tPenaltyExp = 3.0;
    constexpr auto tHeatSourceConst  = 4.0;
    Plato::ScalarVector tResult("result", tNumCells);
    Plato::ScalarMultiVector tControl("control", tNumCells, tNumNodesPerCell);
    auto tHostControl = Kokkos::create_mirror(tControl);
    tHostControl(0,0) = 0.5; tHostControl(0,1) = 0.5; tHostControl(0,2) = 0.5;
    tHostControl(1,0) = 1.0; tHostControl(1,1) = 1.0; tHostControl(1,2) = 1.0;
    Kokkos::deep_copy(tControl, tHostControl);

    // call device function
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        tResult(aCellOrdinal) =
            Plato::Fluids::penalize_heat_source_constant<tNumNodesPerCell>(aCellOrdinal, tHeatSourceConst, tPenaltyExp, tControl);
    }, "unit test penalize_heat_source_constant");

    auto tTol = 1e-4;
    std::vector<Plato::Scalar> tGold = {3.5,0.0};
    auto tHostResult = Kokkos::create_mirror(tResult);
    Kokkos::deep_copy(tHostResult, tResult);
    for (auto &tValue : tGold)
    {
        auto tCell = &tValue - &tGold[0];
        TEST_FLOATING_EQUALITY(tValue, tHostResult(tCell), tTol);
    }
    //Plato::print(tResult, "result");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PenalizeThermalDiffusivity)
{
    // set input data for unit test
    constexpr auto tNumCells = 2;
    constexpr auto tSpaceDims = 2;
    constexpr auto tNumNodesPerCell = tSpaceDims + 1;
    Plato::ScalarVector tResult("result", tNumCells);
    Plato::ScalarMultiVector tControl("control", tNumCells, tNumNodesPerCell);
    auto tHostControl = Kokkos::create_mirror(tControl);
    tHostControl(0,0) = 0.5; tHostControl(0,1) = 0.5; tHostControl(0,2) = 0.5;
    tHostControl(1,0) = 1.0; tHostControl(1,1) = 1.0; tHostControl(1,2) = 1.0;
    Kokkos::deep_copy(tControl, tHostControl);
    constexpr auto tPenaltyExp = 3.0;
    constexpr auto tDiffusivityRatio  = 4.0;

    // call device function
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        tResult(aCellOrdinal) =
            Plato::Fluids::penalize_thermal_diffusivity<tNumNodesPerCell>(aCellOrdinal, tDiffusivityRatio, tPenaltyExp, tControl);
    }, "unit test penalize_thermal_diffusivity");

    auto tTol = 1e-4;
    std::vector<Plato::Scalar> tGold = {3.6250,1.0};
    auto tHostResult = Kokkos::create_mirror(tResult);
    Kokkos::deep_copy(tHostResult, tResult);
    for (auto &tValue : tGold)
    {
        auto tCell = &tValue - &tGold[0];
        TEST_FLOATING_EQUALITY(tValue, tHostResult(tCell), tTol);
    }
    //Plato::print(tResult, "result");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, CalculateFlux)
{
    // set input data for unit test
    constexpr auto tNumCells = 2;
    constexpr auto tSpaceDims = 2;
    constexpr auto tNumNodesPerCell = tSpaceDims + 1;
    Plato::ScalarArray3D tGradient("cell gradient", tNumCells, tNumNodesPerCell, tSpaceDims);
    auto tHostGradient = Kokkos::create_mirror(tGradient);
    tHostGradient(0,0,0) = -1; tHostGradient(0,0,1) = 0;
    tHostGradient(0,1,0) = 1;  tHostGradient(0,1,1) = -1;
    tHostGradient(0,2,0) = 0;  tHostGradient(0,2,1) = 1;
    tHostGradient(1,0,0) = 1;  tHostGradient(1,0,1) = 0;
    tHostGradient(1,1,0) = -1; tHostGradient(1,1,1) = 1;
    tHostGradient(1,2,0) = 0;  tHostGradient(1,2,1) = -1;
    Kokkos::deep_copy(tGradient, tHostGradient);
    Plato::ScalarMultiVector tPrevTemp("previous temperature", tNumCells, tNumNodesPerCell);
    auto tHostPrevTemp = Kokkos::create_mirror(tPrevTemp);
    tHostPrevTemp(0,0) = 1; tHostPrevTemp(0,1) = 12; tHostPrevTemp(0,2) = 3;
    tHostPrevTemp(1,0) = 4; tHostPrevTemp(1,1) = 15; tHostPrevTemp(1,2) = 6;
    Kokkos::deep_copy(tPrevTemp, tHostPrevTemp);
    Plato::ScalarMultiVector tFlux("flux", tNumCells, tSpaceDims);

    // call device function
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        Plato::Fluids::calculate_flux<tNumNodesPerCell,tSpaceDims>(aCellOrdinal, tGradient, tPrevTemp, tFlux);
    }, "unit test calculate_flux");

    auto tTol = 1e-4;
    std::vector<std::vector<Plato::Scalar>> tGold = {{11.0,-9.0}, {-11.0,9.0}};
    auto tHostFlux = Kokkos::create_mirror(tFlux);
    Kokkos::deep_copy(tHostFlux, tFlux);
    for(auto& tGArray : tGold)
    {
        auto tCell = &tGArray - &tGold[0];
        for(auto& tGValue : tGArray)
        {
            auto tDof = &tGValue - &tGArray[0];
            TEST_FLOATING_EQUALITY(tGValue,tHostFlux(tCell,tDof),tTol);
        }
    }
    //Plato::print_array_2D(tFlux, "flux");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, CalculateFluxDivergence)
{
    // build mesh, mesh sets, and spatial domain
    constexpr auto tSpaceDims = 2;
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);

    // set input data for unit test
    auto tNumCells = tMesh->nelems();
    constexpr auto tNumNodesPerCell = tSpaceDims + 1;
    Plato::ScalarVector tCellVolume("cell weight", tNumCells);
    Plato::ScalarArray3D tConfigWS("configuration", tNumCells, tNumNodesPerCell, tSpaceDims);
    Plato::ScalarArray3D tGradient("cell gradient", tNumCells, tNumNodesPerCell, tSpaceDims);
    Plato::ScalarMultiVector tResult("result", tNumCells, tNumNodesPerCell);
    Plato::ScalarMultiVector tFlux("flux", tNumCells, tSpaceDims);
    auto tHostFlux = Kokkos::create_mirror(tFlux);
    tHostFlux(0,0) = 1; tHostFlux(0,1) = 2;
    tHostFlux(1,0) = 3; tHostFlux(1,1) = 4;
    Kokkos::deep_copy(tFlux, tHostFlux);

    // set functors for unit test
    Plato::ComputeGradientWorkset<tSpaceDims> tComputeGradient;
    Plato::NodeCoordinate<tSpaceDims> tNodeCoordinate( (&tMesh.operator*()) );

    // call device function
    Plato::workset_config_scalar<tSpaceDims, tNumNodesPerCell>(tMesh->nelems(), tNodeCoordinate, tConfigWS);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        tComputeGradient(aCellOrdinal, tGradient, tConfigWS, tCellVolume);
        Plato::Fluids::calculate_flux_divergence<tNumNodesPerCell,tSpaceDims>(aCellOrdinal, tGradient, tCellVolume, tFlux, tResult);
    }, "unit test calculate_flux_divergence");

    auto tTol = 1e-4;
    std::vector<std::vector<Plato::Scalar>> tGold = {{-1.0,-1.0,2.0}, {3.0,1.0,-4.0}};
    auto tHostResult = Kokkos::create_mirror(tResult);
    Kokkos::deep_copy(tHostResult, tResult);
    for(auto& tGArray : tGold)
    {
        auto tCell = &tGArray - &tGold[0];
        for(auto& tGValue : tGArray)
        {
            auto tDof = &tGValue - &tGArray[0];
            TEST_FLOATING_EQUALITY(tGValue,tHostResult(tCell,tDof),tTol);
        }
    }
    //Plato::print_array_2D(tResult, "results");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, IntegrateScalarField)
{
    // set input data for unit test
    constexpr auto tNumCells = 2;
    constexpr auto tSpaceDims = 2;
    constexpr auto tNumNodesPerCell = tSpaceDims + 1;
    Plato::ScalarVector tCellVolume("cell weight", tNumCells);
    Plato::blas1::fill(0.5, tCellVolume);
    Plato::ScalarVector tSource("cell source", tNumCells);
    auto tHostSource = Kokkos::create_mirror(tSource);
    tHostSource(0) = 1; tHostSource(1) = 2;
    Kokkos::deep_copy(tSource, tHostSource);
    Plato::ScalarMultiVector tResult("result", tNumCells, tNumNodesPerCell);

    // call device kernel
    Plato::LinearTetCubRuleDegreeOne<tSpaceDims> tCubRule;
    auto tBasisFunctions = tCubRule.getBasisFunctions();
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        Plato::Fluids::integrate_scalar_field<tNumNodesPerCell>(aCellOrdinal, tBasisFunctions, tCellVolume, tSource, tResult);
    }, "unit test integrate_scalar_field");

    auto tTol = 1e-4;
    std::vector<std::vector<Plato::Scalar>> tGold =
        {{0.166666666666667,0.166666666666667,0.166666666666667},
         {0.333333333333333,0.333333333333333,0.333333333333333}};
    auto tHostResult = Kokkos::create_mirror(tResult);
    Kokkos::deep_copy(tHostResult, tResult);
    for(auto& tGArray : tGold)
    {
        auto tCell = &tGArray - &tGold[0];
        for(auto& tGValue : tGArray)
        {
            auto tDof = &tGValue - &tGArray[0];
            TEST_FLOATING_EQUALITY(tGValue,tHostResult(tCell,tDof),tTol);
        }
    }
    //Plato::print_array_2D(tResult, "results");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, CalculateConvectiveForces)
{
    // build mesh, mesh sets, and spatial domain
    constexpr auto tSpaceDims = 2;
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);

    // set input data for unit test
    auto tNumCells = tMesh->nelems();
    constexpr auto tNumNodesPerCell = tSpaceDims + 1;
    Plato::ScalarVector tCellVolume("cell weight", tNumCells);
    Plato::ScalarArray3D tConfigWS("configuration", tNumCells, tNumNodesPerCell, tSpaceDims);
    Plato::ScalarArray3D tGradient("cell gradient", tNumCells, tNumNodesPerCell, tSpaceDims);
    Plato::ScalarMultiVector tPrevTemp("previous temperature", tNumCells, tNumNodesPerCell);
    auto tHostPrevTemp = Kokkos::create_mirror(tPrevTemp);
    tHostPrevTemp(0,0) = 1; tHostPrevTemp(0,1) = 2; tHostPrevTemp(0,2) = 3;
    tHostPrevTemp(1,0) = 4; tHostPrevTemp(1,1) = 5; tHostPrevTemp(1,2) = 6;
    Kokkos::deep_copy(tPrevTemp, tHostPrevTemp);
    Plato::ScalarMultiVector tPrevVelGP("previous velocities", tNumCells, tSpaceDims);
    auto tHostPrevVelGP = Kokkos::create_mirror(tPrevVelGP);
    tHostPrevVelGP(0,0) = 1; tHostPrevVelGP(0,1) = 2;
    tHostPrevVelGP(1,0) = 3; tHostPrevVelGP(1,1) = 4;
    Kokkos::deep_copy(tPrevVelGP, tHostPrevVelGP);
    Plato::ScalarVector tForces("internal force", tNumCells);

    // set functors for unit test
    Plato::ComputeGradientWorkset<tSpaceDims> tComputeGradient;
    Plato::NodeCoordinate<tSpaceDims> tNodeCoordinate( (&tMesh.operator*()) );

    Plato::workset_config_scalar<tSpaceDims, tNumNodesPerCell>(tMesh->nelems(), tNodeCoordinate, tConfigWS);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        tComputeGradient(aCellOrdinal, tGradient, tConfigWS, tCellVolume);
        Plato::Fluids::calculate_convective_forces<tNumNodesPerCell, tSpaceDims>(aCellOrdinal, tGradient, tPrevVelGP, tPrevTemp, tForces);
    }, "unit test calculate_convective_forces");

    auto tTol = 1e-4;
    std::vector<Plato::Scalar> tGold = {3.0,-7.0};
    auto tHostForces = Kokkos::create_mirror(tForces);
    Kokkos::deep_copy(tHostForces, tForces);
    for (auto &tValue : tGold)
    {
        auto tCell = &tValue - &tGold[0];
        TEST_FLOATING_EQUALITY(tValue, tHostForces(tCell), tTol);
    }
    //Plato::print(tForces, "convective forces");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, VelocityCorrectorResidual)
{
    // set xml file inputs
    Teuchos::RCP<Teuchos::ParameterList> tInputs =
        Teuchos::getParametersFromXmlString(
            "<ParameterList name='Plato Problem'>"
            "  <ParameterList name='Hyperbolic'>"
            "    <Parameter name='Heat Transfer' type='string' value='Natural Convection'/>"
            "    <ParameterList  name='Dimensionless Properties'>"
            "      <Parameter  name='Darcy Number'   type='double'        value='1.0'/>"
            "      <Parameter  name='Prandtl Number' type='double'        value='1.0'/>"
            "      <Parameter  name='Grashof Number' type='Array(double)' value='{0.0,1.0}'/>"
            "    </ParameterList>"
            "    <ParameterList name='Momentum Conservation'>"
            "      <ParameterList name='Penalty Function'>"
            "        <Parameter name='Brinkman Convexity Parameter' type='double' value='0.5'/>"
            "      </ParameterList>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList  name='Time Integration'>"
            "    <Parameter name='Artificial Damping Two' type='double' value='0.2'/>"
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
    tHostPrevVel(0) = 1; tHostPrevVel(1) = 2; tHostPrevVel(2) = 3;
    tHostPrevVel(3) = 4; tHostPrevVel(4) = 5; tHostPrevVel(5) = 6;
    Kokkos::deep_copy(tPrevVel, tHostPrevVel);
    tVariables.vector("previous velocity", tPrevVel);
    Plato::ScalarVector tPrevPress("previous pressure", tNumNodes);
    Plato::blas1::fill(1.0, tPrevPress);
    tVariables.vector("previous pressure", tPrevPress);
    Plato::ScalarVector tPrevTemp("previous temperature", tNumNodes);
    tVariables.vector("previous temperature", tPrevTemp);

    Plato::ScalarVector tCurVel("current velocity", tNumVelDofs);
    tVariables.vector("current velocity", tCurVel);
    Plato::ScalarVector tCurPred("current predictor", tNumVelDofs);
    auto tHostCurPred = Kokkos::create_mirror(tCurPred);
    tHostCurPred(0) = 7; tHostCurPred(1) = 8; tHostCurPred(2) = 9;
    tHostCurPred(3) = 10; tHostCurPred(4) = 11; tHostCurPred(5) = 12;
    Kokkos::deep_copy(tCurPred, tHostCurPred);
    tVariables.vector("current predictor", tCurPred);
    Plato::ScalarVector tCurPress("current pressure", tNumNodes);
    auto tHostCurPress = Kokkos::create_mirror(tCurPress);
    tHostCurPress(0) = 1; tHostCurPress(1) = 2;
    tHostCurPress(2) = 3; tHostCurPress(3) = 4;
    Kokkos::deep_copy(tCurPress, tHostCurPress);
    tVariables.vector("current pressure", tCurPress);
    Plato::ScalarVector tCurTemp("current temperature", tNumNodes);
    tVariables.vector("current temperature", tCurTemp);

    Plato::ScalarVector tTimeSteps("time step", 1);
    Plato::blas1::fill(0.1, tTimeSteps);
    tVariables.vector("critical time step", tTimeSteps);

    // allocate vector function
    Plato::DataMap tDataMap;
    std::string tFuncName("Velocity Corrector");
    Plato::Fluids::VectorFunction<PhysicsT> tVectorFunction(tFuncName, tModel, tDataMap, tInputs.operator*());

    // test vector function value
    auto tResidual = tVectorFunction.value(tControls, tVariables);

    auto tTol = 1e-4;
    auto tHostResidual = Kokkos::create_mirror(tResidual);
    Kokkos::deep_copy(tHostResidual, tResidual);
    std::vector<Plato::Scalar> tGold = {-2.48717,-2.77761,-1.4955,-1.66333,-2.4845,-2.77561,-0.9885,-1.11444};
    for(auto& tValue : tGold)
    {
        auto tIndex = &tValue - &tGold[0];
        TEST_FLOATING_EQUALITY(tValue,tHostResidual(tIndex),tTol);
    }
    //Plato::print(tResidual, "residual");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, CalculatePressureGradient)
{
    // build mesh, mesh sets, and spatial domain
    constexpr auto tSpaceDims = 2;
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);

    // set input data for unit test
    auto tNumCells = tMesh->nelems();
    constexpr auto tNumNodesPerCell = tSpaceDims + 1;
    Plato::ScalarVector tCellVolume("cell weight", tNumCells);
    Plato::ScalarArray3D tConfigWS("configuration", tNumCells, tNumNodesPerCell, tSpaceDims);
    Plato::ScalarArray3D tGradient("cell gradient", tNumCells, tNumNodesPerCell, tSpaceDims);
    Plato::ScalarMultiVector tCurPress("current pressure", tNumCells, tNumNodesPerCell);
    auto tHostCurPress = Kokkos::create_mirror(tCurPress);
    tHostCurPress(0,0) = 1; tHostCurPress(0,1) = 2; tHostCurPress(0,2) = 3;
    tHostCurPress(1,0) = 4; tHostCurPress(1,1) = 5; tHostCurPress(1,2) = 6;
    Kokkos::deep_copy(tCurPress, tHostCurPress);
    Plato::ScalarMultiVector tPrevPress("previous pressure", tNumCells, tNumNodesPerCell);
    auto tHostPrevPress = Kokkos::create_mirror(tPrevPress);
    tHostPrevPress(0,0) = 1; tHostPrevPress(0,1) = 12; tHostPrevPress(0,2) = 3;
    tHostPrevPress(1,0) = 4; tHostPrevPress(1,1) = 15; tHostPrevPress(1,2) = 6;
    Kokkos::deep_copy(tPrevPress, tHostPrevPress);
    Plato::ScalarMultiVector tPressGrad("result", tNumCells, tSpaceDims);

    // set functors for unit test
    Plato::ComputeGradientWorkset<tSpaceDims> tComputeGradient;
    Plato::NodeCoordinate<tSpaceDims> tNodeCoordinate( (&tMesh.operator*()) );

    // call device kernel
    auto tTheta = 0.2;
    Plato::workset_config_scalar<tSpaceDims, tNumNodesPerCell>(tMesh->nelems(), tNodeCoordinate, tConfigWS);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        tComputeGradient(aCellOrdinal, tGradient, tConfigWS, tCellVolume);
        Plato::Fluids::calculate_pressure_gradient<tNumNodesPerCell, tSpaceDims>
            (aCellOrdinal, tTheta, tGradient, tCurPress, tPrevPress, tPressGrad);
    }, "unit test calculate_pressure_gradient");

    auto tTol = 1e-4;
    std::vector<std::vector<Plato::Scalar>> tGold = {{9.0,-7.0}, {-9.0,7.0}};
    auto tHostPressGrad = Kokkos::create_mirror(tPressGrad);
    Kokkos::deep_copy(tHostPressGrad, tPressGrad);
    for(auto& tGoldVector : tGold)
    {
        auto tCell = &tGoldVector - &tGold[0];
        for(auto& tGoldValue : tGoldVector)
        {
            auto tDof = &tGoldValue - &tGoldVector[0];
            TEST_FLOATING_EQUALITY(tGoldValue,tHostPressGrad(tCell,tDof),tTol);
        }
    }
    //Plato::print_array_2D(tPressGrad, "pressure gradient");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, CalculateBrinkmanForces)
{
    constexpr auto tNumCells = 2;
    constexpr auto tSpaceDims = 2;
    constexpr auto tBrinkmanCoeff = 0.5;
    Plato::ScalarMultiVector tResult("results", tNumCells, tSpaceDims);
    Plato::ScalarMultiVector tPrevVelGP("previous velocities", tNumCells, tSpaceDims);
    auto tHostPrevVelGP = Kokkos::create_mirror(tPrevVelGP);
    tHostPrevVelGP(0,0) = 1; tHostPrevVelGP(0,1) = 2;
    tHostPrevVelGP(1,0) = 3; tHostPrevVelGP(1,1) = 4;
    Kokkos::deep_copy(tPrevVelGP, tHostPrevVelGP);

    // call device kernel
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        Plato::Fluids::calculate_brinkman_forces<tSpaceDims>(aCellOrdinal, tBrinkmanCoeff, tPrevVelGP, tResult);
    }, "unit test calculate_brinkman_forces");

    auto tTol = 1e-4;
    std::vector<std::vector<Plato::Scalar>> tGold = {{0.5,1.0},{1.5,2.0}};
    auto tHostResult = Kokkos::create_mirror(tResult);
    Kokkos::deep_copy(tHostResult, tResult);
    for(auto& tGoldVector : tGold)
    {
        auto tCell = &tGoldVector - &tGold[0];
        for(auto& tGoldValue : tGoldVector)
        {
            auto tDof = &tGoldValue - &tGoldVector[0];
            TEST_FLOATING_EQUALITY(tGoldValue,tHostResult(tCell,tDof),tTol);
        }
    }
    //Plato::print_array_2D(tResult, "brinkman forces");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, IntegrateStabilizingForces)
{
    // build mesh, mesh sets, and spatial domain
    constexpr auto tSpaceDims = 2;
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);

    // set input data for unit test
    auto tNumCells = tMesh->nelems();
    constexpr auto tNumNodesPerCell = tSpaceDims + 1;
    constexpr auto tNumDofsPerCell = tNumNodesPerCell * tSpaceDims;
    Plato::ScalarVector tCellVolume("cell weight", tNumCells);
    Plato::ScalarVector tDivergence("divergence", tNumCells);
    auto tHostDivergence = Kokkos::create_mirror(tDivergence);
    tHostDivergence(0) = 4; tHostDivergence(1) = -4;
    Kokkos::deep_copy(tDivergence, tHostDivergence);
    Plato::ScalarArray3D tConfigWS("configuration", tNumCells, tNumNodesPerCell, tSpaceDims);
    Plato::ScalarArray3D tGradient("cell gradient", tNumCells, tNumNodesPerCell, tSpaceDims);
    Plato::ScalarMultiVector tPrevVelGP("previous velocities", tNumCells, tSpaceDims);
    auto tHostPrevVelGP = Kokkos::create_mirror(tPrevVelGP);
    tHostPrevVelGP(0,0) = 1; tHostPrevVelGP(0,1) = 2;
    tHostPrevVelGP(1,0) = 3; tHostPrevVelGP(1,1) = 4;
    Kokkos::deep_copy(tPrevVelGP, tHostPrevVelGP);
    Plato::ScalarMultiVector tForce("internal force", tNumCells, tSpaceDims);
    Plato::blas2::fill(1.0,tForce);
    Plato::ScalarMultiVector tResult("result", tNumCells, tNumDofsPerCell);

    // set functors for unit test
    Plato::LinearTetCubRuleDegreeOne<tSpaceDims> tCubRule;
    Plato::ComputeGradientWorkset<tSpaceDims> tComputeGradient;
    Plato::NodeCoordinate<tSpaceDims> tNodeCoordinate( (&tMesh.operator*()) );

    // call device kernel
    auto tCubWeight = tCubRule.getCubWeight();
    auto tBasisFunctions = tCubRule.getBasisFunctions();
    Plato::workset_config_scalar<tSpaceDims, tNumNodesPerCell>(tMesh->nelems(), tNodeCoordinate, tConfigWS);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        tComputeGradient(aCellOrdinal, tGradient, tConfigWS, tCellVolume);
        tCellVolume(aCellOrdinal) *= tCubWeight;
        Plato::Fluids::integrate_stabilizing_vector_force<tNumNodesPerCell, tSpaceDims>
            (aCellOrdinal, tCellVolume, tGradient, tPrevVelGP, tForce, tResult);
    }, "unit test integrate_stabilizing_vector_force");

    // TODO: FIX GOLD VALUES ONCE THINGS ARE WORKING
    auto tTol = 1e-4;
    std::vector<std::vector<Plato::Scalar>> tGold =
        {{0.166666666666667,0.166666666666667,0.166666666666667,0.166666666666667,1.666666666666667,1.666666666666667},
         {0.833333333333333,0.833333333333333,-0.166666666666667,-0.166666666666667,-2.666666666666667,-2.666666666666667}};
    auto tHostResult = Kokkos::create_mirror(tResult);
    Kokkos::deep_copy(tHostResult, tResult);
    for(auto& tGoldVector : tGold)
    {
        auto tCell = &tGoldVector - &tGold[0];
        for(auto& tGoldValue : tGoldVector)
        {
            auto tDof = &tGoldValue - &tGoldVector[0];
            TEST_FLOATING_EQUALITY(tGoldValue,tHostResult(tCell,tDof),tTol);
        }
    }
    //Plato::print_array_2D(tResult, "stabilizing forces");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, CalculateInertialForces)
{
    // set input data for unit test
    constexpr auto tNumCells = 2;
    constexpr auto tSpaceDims = 2;
    constexpr auto tNumNodesPerCell = tSpaceDims + 1;
    constexpr auto tNumDofsPerCell = tNumNodesPerCell * tSpaceDims;
    Plato::ScalarVector tCellVolume("cell weight", tNumCells);
    Plato::blas1::fill(0.5, tCellVolume);
    Plato::ScalarMultiVector tResult("result", tNumCells, tNumDofsPerCell);
    Plato::ScalarMultiVector tPredVelGP("predicted velocities", tNumCells, tSpaceDims);
    Plato::ScalarMultiVector tPrevVelGP("previous velocities", tNumCells, tSpaceDims);
    auto tHostPrevVelGP = Kokkos::create_mirror(tPrevVelGP);
    tHostPrevVelGP(0,0) = 1; tHostPrevVelGP(0,1) = 2;
    tHostPrevVelGP(1,0) = 3; tHostPrevVelGP(1,1) = 4;
    Kokkos::deep_copy(tPrevVelGP, tHostPrevVelGP);

    Plato::LinearTetCubRuleDegreeOne<tSpaceDims> tCubRule;
    auto tBasisFunctions = tCubRule.getBasisFunctions();
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        Plato::Fluids::integrate_momentum_inertial_forces<tNumNodesPerCell, tSpaceDims>
            (aCellOrdinal, tBasisFunctions, tCellVolume, tPredVelGP, tPrevVelGP, tResult);
    }, "unit test integrate_momentum_inertial_forces");

    auto tTol = 1e-4;
    std::vector<std::vector<Plato::Scalar>> tGold =
        {{-1.666667e-01,-3.333333e-01,-1.666667e-01,-3.333333e-01,-1.666667e-01,-3.333333e-01},
         {-5.000000e-01,-6.666667e-01,-5.000000e-01,-6.666667e-01,-5.000000e-01,-6.666667e-01}};
    auto tHostResult = Kokkos::create_mirror(tResult);
    Kokkos::deep_copy(tHostResult, tResult);
    for(auto& tGoldVector : tGold)
    {
        auto tCell = &tGoldVector - &tGold[0];
        for(auto& tGoldValue : tGoldVector)
        {
            auto tDof = &tGoldValue - &tGoldVector[0];
            TEST_FLOATING_EQUALITY(tGoldValue,tHostResult(tCell,tDof),tTol);
        }
    }
    //Plato::print_array_2D(tResult, "inertial forces");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, Integrate)
{
    // set input data for unit test
    constexpr auto tNumCells = 2;
    constexpr auto tSpaceDims = 2;
    constexpr auto tNumNodesPerCell = tSpaceDims + 1;
    constexpr auto tNumDofsPerCell = tNumNodesPerCell * tSpaceDims;
    Plato::ScalarVector tCellVolume("cell weight", tNumCells);
    Plato::blas1::fill(0.5, tCellVolume);
    Plato::ScalarMultiVector tResult("results", tNumCells, tNumDofsPerCell);
    Plato::ScalarMultiVector tInternalForces("internal forces", tNumCells, tSpaceDims);
    auto tHostInternalForces = Kokkos::create_mirror(tInternalForces);
    tHostInternalForces(0,0) = 26.0 ; tHostInternalForces(0,1) = 30.0;
    tHostInternalForces(1,0) = -74.0; tHostInternalForces(1,1) = -78.0;
    Kokkos::deep_copy(tInternalForces, tHostInternalForces);

    // call device kernel
    Plato::LinearTetCubRuleDegreeOne<tSpaceDims> tCubRule;
    auto tCubWeight = tCubRule.getCubWeight();
    auto tBasisFunctions = tCubRule.getBasisFunctions();
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        Plato::Fluids::integrate_vector_field<tNumNodesPerCell, tSpaceDims>
            (aCellOrdinal, tBasisFunctions, tCellVolume, tInternalForces, tResult);
    }, "unit test integrate_vector_field");

    auto tTol = 1e-4;
    std::vector<std::vector<Plato::Scalar>> tGold =
        {{4.333333,5.0,4.333333,5.0,4.333333,5.0},{-12.33333,-13.0,-12.33333,-13.0,-12.33333,-13.0}};
    auto tHostResult = Kokkos::create_mirror(tResult);
    Kokkos::deep_copy(tHostResult, tResult);
    for(auto& tGoldVector : tGold)
    {
        auto tCell = &tGoldVector - &tGold[0];
        for(auto& tGoldValue : tGoldVector)
        {
            auto tDof = &tGoldValue - &tGoldVector[0];
            TEST_FLOATING_EQUALITY(tGoldValue,tHostResult(tCell,tDof),tTol);
        }
    }
    //Plato::print_array_2D(tResult, "integrated forces");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, CalculateAdvectedInternalForces)
{
    // build mesh, mesh sets, and spatial domain
    constexpr auto tSpaceDims = 2;
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);

    // set input data for unit test
    auto tNumCells = tMesh->nelems();
    constexpr auto tNumVelDofsPerNode = tSpaceDims;
    constexpr auto tNumNodesPerCell = tSpaceDims + 1;
    constexpr auto tNumDofsPerCell = tNumNodesPerCell * tSpaceDims;
    Plato::ScalarVector tCellVolume("cell weight", tNumCells);
    Plato::ScalarArray3D tConfigWS("configuration", tNumCells, tNumNodesPerCell, tSpaceDims);
    Plato::ScalarArray3D tGradient("cell gradient", tNumCells, tNumNodesPerCell, tSpaceDims);
    Plato::ScalarMultiVector tPrevVelGP("previous velocity at GP", tNumCells, tSpaceDims);
    Plato::ScalarMultiVector tPrevVelWS("previous velocity", tNumCells, tNumDofsPerCell);
    Plato::ScalarMultiVector tInternalForces("internal forces", tNumCells, tSpaceDims);
    auto tHostPrevVelWS = Kokkos::create_mirror(tPrevVelWS);
    tHostPrevVelWS(0,0) = 1; tHostPrevVelWS(0,1) = 2; tHostPrevVelWS(0,2) = 3; tHostPrevVelWS(0,3) = 4 ; tHostPrevVelWS(0,4) = 5 ; tHostPrevVelWS(0,5) = 6;
    tHostPrevVelWS(1,0) = 7; tHostPrevVelWS(1,1) = 8; tHostPrevVelWS(1,2) = 9; tHostPrevVelWS(1,3) = 10; tHostPrevVelWS(1,4) = 11; tHostPrevVelWS(1,5) = 12;
    Kokkos::deep_copy(tPrevVelWS, tHostPrevVelWS);

    // set functors for unit test
    Plato::LinearTetCubRuleDegreeOne<tSpaceDims> tCubRule;
    Plato::ComputeGradientWorkset<tSpaceDims> tComputeGradient;
    Plato::NodeCoordinate<tSpaceDims> tNodeCoordinate( (&tMesh.operator*()) );
    Plato::InterpolateFromNodal<tSpaceDims, tNumVelDofsPerNode, 0, tSpaceDims> tIntrplVectorField;

    // call device kernel
    auto tCubWeight = tCubRule.getCubWeight();
    auto tBasisFunctions = tCubRule.getBasisFunctions();
    Plato::workset_config_scalar<tSpaceDims, tNumNodesPerCell>(tMesh->nelems(), tNodeCoordinate, tConfigWS);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        tComputeGradient(aCellOrdinal, tGradient, tConfigWS, tCellVolume);
        tCellVolume(aCellOrdinal) *= tCubWeight;

        tIntrplVectorField(aCellOrdinal, tBasisFunctions, tPrevVelWS, tPrevVelGP);
        Plato::Fluids::calculate_advected_momentum_forces<tNumNodesPerCell, tSpaceDims>
            (aCellOrdinal, tGradient, tPrevVelWS, tPrevVelGP, tInternalForces);
    }, "unit test calculate_advected_momentum_forces");

    auto tTol = 1e-4;
    std::vector<std::vector<Plato::Scalar>> tGold = {{26.0,30.0},{-74.0,-78.0}};
    auto tHostInternalForces = Kokkos::create_mirror(tInternalForces);
    Kokkos::deep_copy(tHostInternalForces, tInternalForces);
    for(auto& tGoldVector : tGold)
    {
        auto tVecIndex = &tGoldVector - &tGold[0];
        for(auto& tGoldValue : tGoldVector)
        {
            auto tValIndex = &tGoldValue - &tGoldVector[0];
            TEST_FLOATING_EQUALITY(tGoldValue,tHostInternalForces(tVecIndex,tValIndex),tTol);
        }
    }
    //Plato::print_array_2D(tInternalForces, "advected internal forces");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, CalculateNaturalConvectiveForces)
{
    // set input data for unit test
    constexpr auto tNumCells = 2;
    constexpr auto tSpaceDims = 2;
    Plato::ScalarVector tCellVolume("cell weight", tNumCells);
    Plato::ScalarVector tPrevTempGP("temperature at GP", tNumCells);
    Plato::blas1::fill(1.0, tPrevTempGP);
    Plato::ScalarMultiVector tResultGP("cell stabilized convective forces", tNumCells, tSpaceDims);
    Plato::Scalar tPenalizedPrNumTimesPrNum = 0.25;
    Plato::ScalarVector tPenalizedGrNum("Grashof Number", tSpaceDims);
    auto tHostPenalizedGrNum = Kokkos::create_mirror(tPenalizedGrNum);
    tHostPenalizedGrNum(1) = 1.0;
    Kokkos::deep_copy(tPenalizedGrNum, tHostPenalizedGrNum);

    // call device kernel
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        Plato::Fluids::calculate_natural_convective_forces<tSpaceDims>
            (aCellOrdinal, tPenalizedPrNumTimesPrNum, tPenalizedGrNum, tPrevTempGP, tResultGP);
    }, "unit test calculate_natural_convective_forces");

    auto tTol = 1e-4;
    std::vector<std::vector<Plato::Scalar>> tGold = {{0.0,0.25},{0.0,0.25}};
    auto tHostResultGP = Kokkos::create_mirror(tResultGP);
    Kokkos::deep_copy(tHostResultGP, tResultGP);
    for(auto& tGoldVector : tGold)
    {
        auto tVecIndex = &tGoldVector - &tGold[0];
        for(auto& tGoldValue : tGoldVector)
        {
            auto tValIndex = &tGoldValue - &tGoldVector[0];
            TEST_FLOATING_EQUALITY(tGoldValue,tHostResultGP(tVecIndex,tValIndex),tTol);
        }
    }
    //Plato::print_array_2D(tResultWS, "stabilized natural convective forces");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, IntegrateViscousForces)
{
    // build mesh, mesh sets, and spatial domain
    constexpr auto tSpaceDims = 2;
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);

    // set input data for unit test
    auto tNumCells = tMesh->nelems();
    constexpr auto tNumNodesPerCell = tSpaceDims + 1;
    constexpr auto tNumDofsPerCell = tNumNodesPerCell * tSpaceDims;
    Plato::ScalarVector tCellVolume("cell weight", tNumCells);
    Plato::ScalarArray3D tStrainRate("cell strain rate", tNumCells, tSpaceDims, tSpaceDims);
    Plato::ScalarArray3D tConfigWS("configuration", tNumCells, tNumNodesPerCell, tSpaceDims);
    Plato::ScalarArray3D tGradient("cell gradient", tNumCells, tNumNodesPerCell, tSpaceDims);
    Plato::ScalarMultiVector tResultWS("cell viscous forces", tNumCells, tNumDofsPerCell);
    Plato::ScalarMultiVector tPrevVelWS("previous velocity workset", tNumCells, tNumDofsPerCell);
    auto tHostPrevVelWS = Kokkos::create_mirror(tPrevVelWS);
    tHostPrevVelWS(0,0) = 1; tHostPrevVelWS(0,1) = 2; tHostPrevVelWS(0,2) = 3; tHostPrevVelWS(0,3) = 4 ; tHostPrevVelWS(0,4) = 5 ; tHostPrevVelWS(0,5) = 6;
    tHostPrevVelWS(1,0) = 7; tHostPrevVelWS(1,1) = 8; tHostPrevVelWS(1,2) = 9; tHostPrevVelWS(1,3) = 10; tHostPrevVelWS(1,4) = 11; tHostPrevVelWS(1,5) = 12;
    Kokkos::deep_copy(tPrevVelWS, tHostPrevVelWS);
    Plato::Scalar tPenalizedPrNum = 0.5;

    // set functors for unit test
    Plato::LinearTetCubRuleDegreeOne<tSpaceDims> tCubRule;
    Plato::ComputeGradientWorkset<tSpaceDims> tComputeGradient;
    Plato::NodeCoordinate<tSpaceDims> tNodeCoordinate( (&tMesh.operator*()) );

    // call device kernel
    auto tCubWeight = tCubRule.getCubWeight();
    Plato::workset_config_scalar<tSpaceDims, tNumNodesPerCell>(tMesh->nelems(), tNodeCoordinate, tConfigWS);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        tComputeGradient(aCellOrdinal, tGradient, tConfigWS, tCellVolume);
        tCellVolume(aCellOrdinal) *= tCubWeight;

        Plato::Fluids::strain_rate<tNumNodesPerCell, tSpaceDims>
            (aCellOrdinal, tPrevVelWS, tGradient, tStrainRate);
        Plato::Fluids::integrate_viscous_forces<tNumNodesPerCell, tSpaceDims>
            (aCellOrdinal, tPenalizedPrNum, tCellVolume, tGradient, tStrainRate, tResultWS);
    }, "unit test integrate_viscous_forces");

    auto tTol = 1e-4;
    std::vector<std::vector<Plato::Scalar>> tGold = {{-1.0,-1.0,0.0,0.0,1.0,1.0},{-1.0,-1.0,0.0,0.0,1.0,1.0}};
    auto tHostResultWS = Kokkos::create_mirror(tResultWS);
    Kokkos::deep_copy(tHostResultWS, tResultWS);
    for(auto& tGoldVector : tGold)
    {
        auto tVecIndex = &tGoldVector - &tGold[0];
        for(auto& tGoldValue : tGoldVector)
        {
            auto tValIndex = &tGoldValue - &tGoldVector[0];
            TEST_FLOATING_EQUALITY(tGoldValue,tHostResultWS(tVecIndex,tValIndex),tTol);
        }
    }
    //Plato::print_array_2D(tResultWS, "viscous forces");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, BLAS1_update)
{
    constexpr auto tNumCells = 2;
    constexpr auto tNumDofsPerCell = 6;
    Plato::ScalarMultiVector tVec1("vector one", tNumCells, tNumDofsPerCell);
    Plato::blas2::fill(1.0, tVec1);
    Plato::ScalarMultiVector tVec2("vector two", tNumCells, tNumDofsPerCell);
    Plato::blas2::fill(2.0, tVec2);

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        auto tConstant = static_cast<Plato::Scalar>(aCellOrdinal);
        Plato::blas1::update<tNumDofsPerCell>(aCellOrdinal, 2.0, tVec1, 3.0 + tConstant, tVec2);
    },"device_blas1_update");

    auto tTol = 1e-4;
    auto tHostVec2 = Kokkos::create_mirror(tVec2);
    Kokkos::deep_copy(tHostVec2, tVec2);
    std::vector<std::vector<Plato::Scalar>> tGold = { {8.0, 8.0, 8.0, 8.0, 8.0, 8.0}, {10.0, 10.0, 10.0, 10.0, 10.0, 10.0} };
    for(auto& tVector : tGold)
    {
        auto tCell = &tVector - &tGold[0];
        for(auto& tValue : tVector)
        {
            auto tDim = &tValue - &tVector[0];
            TEST_FLOATING_EQUALITY(tValue, tHostVec2(tCell, tDim), tTol);
        }
    }
}

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

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, NaturalConvectionBrinkman)
{
    // set xml file inputs
    Teuchos::RCP<Teuchos::ParameterList> tInputs =
        Teuchos::getParametersFromXmlString(
            "<ParameterList name='Plato Problem'>"
            "  <ParameterList name='Hyperbolic'>"
            "    <Parameter name='Scenario' type='string' value='Density TO'/>"
            "    <Parameter name='Heat Transfer' type='string' value='Natural'/>"
            "    <ParameterList  name='Dimensionless Properties'>"
            "      <Parameter  name='Darcy Number'   type='double'        value='1.0'/>"
            "      <Parameter  name='Prandtl Number' type='double'        value='1.0'/>"
            "      <Parameter  name='Grashof Number' type='Array(double)' value='{0.0,1.0}'/>"
            "    </ParameterList>"
            "    <ParameterList name='Momentum Conservation'>"
            "      <ParameterList name='Penalty Function'>"
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

    Plato::ScalarVector tTimeSteps("time step", 1);
    Plato::blas1::fill(0.1, tTimeSteps);
    tVariables.vector("critical time step", tTimeSteps);

    // allocate vector function
    Plato::DataMap tDataMap;
    std::string tFuncName("Velocity Predictor");
    Plato::Fluids::VectorFunction<PhysicsT> tVectorFunction(tFuncName, tModel, tDataMap, tInputs.operator*());

    // test vector function value
    auto tResidual = tVectorFunction.value(tControls, tVariables);

    auto tHostResidual = Kokkos::create_mirror(tResidual);
    Kokkos::deep_copy(tHostResidual, tResidual);
    std::vector<Plato::Scalar> tGold =
        {-0.212591, -0.190382, -0.163095, -0.161822, -0.293077, -0.0450344, -0.269574, -0.0480922};
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
    auto tFaceOrdinalsOnBoundaryOne = Plato::find_entities_on_non_prescribed_boundary<Omega_h::EDGE, Omega_h::SIDE_SET>(tNames, tDomain.Mesh, tDomain.MeshSets);
    TEST_EQUALITY(0, tFaceOrdinalsOnBoundaryOne.size());

    // test 2
    tNames = {"x+","y+","y-"};
    auto tFaceOrdinalsOnBoundaryTwo = Plato::find_entities_on_non_prescribed_boundary<Omega_h::EDGE, Omega_h::SIDE_SET>(tNames, tDomain.Mesh, tDomain.MeshSets);
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
    auto tFaceOrdinalsOnBoundaryThree = Plato::find_entities_on_non_prescribed_boundary<Omega_h::EDGE, Omega_h::SIDE_SET>(tNames, tDomain.Mesh, tDomain.MeshSets);
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
    auto tFaceOrdinalsOnBoundaryFour = Plato::find_entities_on_non_prescribed_boundary<Omega_h::EDGE, Omega_h::SIDE_SET>(tNames, tDomain.Mesh, tDomain.MeshSets);
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
    auto tFaceOrdinalsOnBoundaryFive =
            Plato::find_entities_on_non_prescribed_boundary<Omega_h::EDGE, Omega_h::SIDE_SET>(tNames, tDomain.Mesh, tDomain.MeshSets);
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
        Plato::Fluids::strain_rate<tNumNodesPerCell, tNumSpaceDims>(aCellOrdinal, tVelocity, tGradient, tStrainRate);
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

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, BLAS1_DeviceScale_Version2)
{
    constexpr Plato::OrdinalType tNumCells = 2;
    constexpr Plato::OrdinalType tNumSpaceDims = 2;
    Plato::ScalarMultiVector tInput("input", tNumCells, tNumSpaceDims);
    Plato::blas2::fill(1.0, tInput);

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        Plato::blas1::scale<tNumSpaceDims>(aCellOrdinal, 4.0, tInput);
    }, "device blas1::scale");

    auto tTol = 1e-6;
    auto tHostInput = Kokkos::create_mirror(tInput);
    Kokkos::deep_copy(tHostInput, tInput);
    for (Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++)
    {
        for (Plato::OrdinalType tDim = 0; tDim < tNumSpaceDims; tDim++)
        {
            TEST_FLOATING_EQUALITY(4.0, tHostInput(tCell, tDim), tTol);
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
            Plato::Fluids::brinkman_penalization<tNumNodesPerCell>(aCellOrdinal, tPhysicalNum, tConvexityParam, tControlWS);
    }, "brinkman_penalization unit test");

    auto tTol = 1e-6;
    auto tHostOutput = Kokkos::create_mirror(tOutput);
    Kokkos::deep_copy(tHostOutput, tOutput);
    for(Plato::OrdinalType tIndex = 0; tIndex < tOutput.size(); tIndex++)
    {
        TEST_FLOATING_EQUALITY(0.4, tHostOutput(tIndex), tTol);
    }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, InternalDissipationEnergyIncompressible_Value)
{
    // set inputs
    Teuchos::RCP<Teuchos::ParameterList> tInputs =
        Teuchos::getParametersFromXmlString(
            "<ParameterList name='Plato Problem'>"
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
    Plato::ScalarVector tTimeSteps("critical time step", 1);
    Plato::blas1::fill(0.01, tTimeSteps);
    tPrimal.vector("critical time step", tTimeSteps);

    // set physics type
    constexpr Plato::OrdinalType tNumSpaceDim = 2;
    using PhysicsT = Plato::IncompressibleFluids<tNumSpaceDim>;

    // build criterion
    Plato::DataMap tDataMap;
    std::string tFuncName("My Criteria");
    Plato::Fluids::ScalarFunction<PhysicsT>
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
            "<ParameterList name='Plato Problem'>"
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
    Plato::ScalarVector tTimeSteps("critical time step", 1);
    Plato::blas1::fill(0.01, tTimeSteps);
    tPrimal.vector("critical time step", tTimeSteps);

    // set physics type
    constexpr Plato::OrdinalType tNumSpaceDim = 2;
    using PhysicsT = Plato::IncompressibleFluids<tNumSpaceDim>;

    // build criterion
    Plato::DataMap tDataMap;
    std::string tFuncName("My Criteria");
    Plato::Fluids::ScalarFunction<PhysicsT>
        tCriterion(tModel, tDataMap, tInputs.operator*(), tFuncName);
    TEST_EQUALITY("My Criteria", tCriterion.name());

    // test criterion value
    auto tTol = 1e-4;
    auto tGradient = tCriterion.gradientControl(tControl, tPrimal);
    auto tHostGradient = Kokkos::create_mirror(tGradient);
    Kokkos::deep_copy(tHostGradient, tGradient);

    std::vector<Plato::Scalar> tGold = {-0.00350222, 0.0, -0.00350222, -0.00350222};
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
            "<ParameterList name='Plato Problem'>"
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
    Plato::ScalarVector tTimeSteps("critical time step", 1);
    Plato::blas1::fill(0.01, tTimeSteps);
    tPrimal.vector("critical time step", tTimeSteps);

    // set physics type
    constexpr Plato::OrdinalType tNumSpaceDim = 2;
    using PhysicsT = Plato::IncompressibleFluids<tNumSpaceDim>;

    // build criterion
    Plato::DataMap tDataMap;
    std::string tFuncName("My Criteria");
    Plato::Fluids::ScalarFunction<PhysicsT>
        tCriterion(tModel, tDataMap, tInputs.operator*(), tFuncName);
    TEST_EQUALITY("My Criteria", tCriterion.name());

    // test criterion value
    auto tTol = 1e-4;
    auto tGradient = tCriterion.gradientConfig(tControl, tPrimal);
    auto tHostGradient = Kokkos::create_mirror(tGradient);
    Kokkos::deep_copy(tHostGradient, tGradient);

    std::vector<Plato::Scalar> tGold = {0.377522, 0.2091, -0.2091, -0.1145};
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
            "<ParameterList name='Plato Problem'>"
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
    Plato::ScalarVector tTimeSteps("critical time step", 1);
    Plato::blas1::fill(0.01, tTimeSteps);
    tPrimal.vector("critical time step", tTimeSteps);

    // set physics type
    constexpr Plato::OrdinalType tNumSpaceDim = 2;
    using PhysicsT = Plato::IncompressibleFluids<tNumSpaceDim>;

    // build criterion
    Plato::DataMap tDataMap;
    std::string tFuncName("My Criteria");
    Plato::Fluids::ScalarFunction<PhysicsT>
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
            "<ParameterList name='Plato Problem'>"
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
    Plato::ScalarVector tTimeSteps("critical time step", 1);
    Plato::blas1::fill(0.01, tTimeSteps);
    tPrimal.vector("critical time step", tTimeSteps);

    // set physics type
    constexpr Plato::OrdinalType tNumSpaceDim = 2;
    using PhysicsT = Plato::IncompressibleFluids<tNumSpaceDim>;

    // build criterion
    Plato::DataMap tDataMap;
    std::string tFuncName("My Criteria");
    Plato::Fluids::ScalarFunction<PhysicsT>
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
            "<ParameterList name='Plato Problem'>"
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
    Plato::ScalarVector tTimeSteps("critical time step", 1);
    Plato::blas1::fill(0.01, tTimeSteps);
    tPrimal.vector("critical time step", tTimeSteps);

    // set physics type
    constexpr Plato::OrdinalType tNumSpaceDim = 2;
    using PhysicsT = Plato::IncompressibleFluids<tNumSpaceDim>;

    // build criterion
    Plato::DataMap tDataMap;
    std::string tFuncName("My Criteria");
    Plato::Fluids::ScalarFunction<PhysicsT>
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
            "<ParameterList name='Plato Problem'>"
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
    Plato::ScalarVector tTimeSteps("critical time step", 1);
    Plato::blas1::fill(0.01, tTimeSteps);
    tPrimal.vector("critical time step", tTimeSteps);

    // set physics type
    constexpr Plato::OrdinalType tNumSpaceDim = 2;
    using PhysicsT = Plato::IncompressibleFluids<tNumSpaceDim>;

    // build criterion
    Plato::DataMap tDataMap;
    std::string tFuncName("My Criteria");
    Plato::Fluids::ScalarFunction<PhysicsT>
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
    using ResidualEvalT = Plato::Fluids::Evaluation<PhysicsT::SimplexT>::Residual;

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
    Plato::ScalarVector tTimeSteps("critical time step", 1);
    Plato::blas1::fill(4.0, tTimeSteps);
    tPrimal.vector("critical time step", tTimeSteps);

    // call build_scalar_function_worksets
    Plato::WorkSets tWorkSets;
    Plato::LocalOrdinalMaps<PhysicsT> tOrdinalMaps(*tMesh);
    Plato::Fluids::build_scalar_function_worksets<ResidualEvalT>
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
    auto tTimeStepWS = Plato::metadata<Plato::ScalarVector>(tWorkSets.get("critical time step"));
    TEST_EQUALITY(1, tTimeStepWS.extent(0));
    auto tHostTimeStepWS = Kokkos::create_mirror(tTimeStepWS);
    Kokkos::deep_copy(tHostTimeStepWS, tTimeStepWS);
    TEST_FLOATING_EQUALITY(4.0, tHostTimeStepWS(0), tTol);

    // test controls results
    auto tNumNodesPerCell = PhysicsT::mNumNodesPerCell;
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
    using ResidualEvalT = Plato::Fluids::Evaluation<PhysicsT::SimplexT>::Residual;

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
    Plato::ScalarVector tTimeSteps("critical time step", 1);
    Plato::blas1::fill(4.0, tTimeSteps);
    tPrimal.vector("critical time step", tTimeSteps);
    Plato::ScalarVector tArtCompress("artificial compressibility", tNumNodes);
    Plato::blas1::fill(5.0, tArtCompress);
    tPrimal.vector("artificial compressibility", tArtCompress);

    // call build_vector_function_worksets
    Plato::WorkSets tWorkSets;
    Plato::LocalOrdinalMaps<PhysicsT> tOrdinalMaps(*tMesh);
    Plato::Fluids::build_vector_function_worksets<ResidualEvalT>
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
    auto tTimeStepWS = Plato::metadata<Plato::ScalarVector>(tWorkSets.get("critical time step"));
    TEST_EQUALITY(1, tTimeStepWS.extent(0));
    auto tHostTimeStepWS = Kokkos::create_mirror(tTimeStepWS);
    Kokkos::deep_copy(tHostTimeStepWS, tTimeStepWS);
    TEST_FLOATING_EQUALITY(4.0, tHostTimeStepWS(0), tTol);

    // test artificial compressibility results
    auto tNumNodesPerCell = PhysicsT::mNumNodesPerCell;
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
    using ResidualEvalT = Plato::Fluids::Evaluation<PhysicsT::SimplexT>::Residual;

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
    Plato::ScalarVector tTimeSteps("critical time step", 1);
    Plato::blas1::fill(4.0, tTimeSteps);
    tPrimal.vector("critical time step", tTimeSteps);
    Plato::ScalarVector tArtCompress("artificial compressibility", tNumNodes);
    Plato::blas1::fill(5.0, tArtCompress);
    tPrimal.vector("artificial compressibility", tArtCompress);

    // set ordinal maps;
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);
    Plato::LocalOrdinalMaps<PhysicsT> tOrdinalMaps(*tMesh);

    // call build_vector_function_worksets
    Plato::WorkSets tWorkSets;
    Plato::Fluids::build_vector_function_worksets<ResidualEvalT>
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
    auto tTimeStepWS = Plato::metadata<Plato::ScalarVector>(tWorkSets.get("critical time step"));
    TEST_EQUALITY(1, tTimeStepWS.extent(0));
    auto tHostTimeStepWS = Kokkos::create_mirror(tTimeStepWS);
    Kokkos::deep_copy(tHostTimeStepWS, tTimeStepWS);
    TEST_FLOATING_EQUALITY(4.0, tHostTimeStepWS(0), tTol);

    // test artificial compressibility results
    auto tNumNodesPerCell = PhysicsT::mNumNodesPerCell;
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
    using ResidualEvalT = Plato::Fluids::Evaluation<PhysicsT::SimplexT>::Residual;

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
    Plato::ScalarVector tTimeSteps("critical time step", 1);
    Plato::blas1::fill(4.0, tTimeSteps);
    tPrimal.vector("critical time step", tTimeSteps);

    // set ordinal maps;
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);
    Plato::LocalOrdinalMaps<PhysicsT> tOrdinalMaps(*tMesh);

    // call build_vector_function_worksets
    Plato::WorkSets tWorkSets;
    Plato::Fluids::build_vector_function_worksets<ResidualEvalT>
        (tNumCells, tControls, tPrimal, tOrdinalMaps, tWorkSets);
    TEST_EQUALITY(tWorkSets.defined("artifical compressibility"), false);
}


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, BuildScalarFunctionWorksets)
{
    constexpr Plato::OrdinalType tNumSpaceDim = 2;
    using PhysicsT = Plato::IncompressibleFluids<tNumSpaceDim>;
    using ResidualEvalT = Plato::Fluids::Evaluation<PhysicsT::SimplexT>::Residual;

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
    Plato::ScalarVector tTimeSteps("critical time step", 1);
    Plato::blas1::fill(4.0, tTimeSteps);
    tPrimal.vector("critical time step", tTimeSteps);

    // set ordinal maps;
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);
    Plato::LocalOrdinalMaps<PhysicsT> tOrdinalMaps(*tMesh);

    // call build_scalar_function_worksets
    Plato::WorkSets tWorkSets;
    Plato::Fluids::build_scalar_function_worksets<ResidualEvalT>
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
    auto tTimeStepWS = Plato::metadata<Plato::ScalarVector>(tWorkSets.get("critical time step"));
    TEST_EQUALITY(1u, tTimeStepWS.extent(0));
    auto tHostTimeStepWS = Kokkos::create_mirror(tTimeStepWS);
    Kokkos::deep_copy(tHostTimeStepWS, tTimeStepWS);
    TEST_FLOATING_EQUALITY(4.0, tHostTimeStepWS(0), tTol);

    // test controls results
    auto tNumNodesPerCell = PhysicsT::mNumNodesPerCell;
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
    Plato::ScalarMultiVector tScalarFieldOrdinals("scalar field ordinals", tNumCells, PhysicsT::mNumNodesPerCell);
    Plato::ScalarArray3D tVectorFieldOrdinals("vector field ordinals", tNumCells, PhysicsT::mNumNodesPerCell, PhysicsT::mNumMomentumDofsPerNode);

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        for(Plato::OrdinalType tNode = 0; tNode < PhysicsT::mNumNodesPerCell; tNode++)
        {
            for(Plato::OrdinalType tDim = 0; tDim < PhysicsT::mNumSpatialDims; tDim++)
            {
                tCoords(aCellOrdinal, tNode, tDim) = tLocalOrdinalMaps.mNodeCoordinate(aCellOrdinal, tNode, tDim);
                tVectorFieldOrdinals(aCellOrdinal, tNode, tDim) = tLocalOrdinalMaps.mVectorFieldOrdinalsMap(aCellOrdinal, tNode, tDim);
            }
        }

        for(Plato::OrdinalType tNode = 0; tNode < PhysicsT::mNumNodesPerCell; tNode++)
        {
            for(Plato::OrdinalType tDim = 0; tDim < PhysicsT::mNumControlDofsPerNode; tDim++)
            {
                tControlOrdinals(aCellOrdinal, tNode) = tLocalOrdinalMaps.mControlOrdinalsMap(aCellOrdinal, tNode, tDim);
                tScalarFieldOrdinals(aCellOrdinal, tNode) = tLocalOrdinalMaps.mScalarFieldOrdinalsMap(aCellOrdinal, tNode, tDim);
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
    auto tHostVectorFieldOrdinals = Kokkos::create_mirror(tVectorFieldOrdinals);
    Kokkos::deep_copy(tHostVectorFieldOrdinals, tVectorFieldOrdinals);

    auto tTol = 1e-6;
    for(Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++)
    {
        for(Plato::OrdinalType tNode = 0; tNode < PhysicsT::mNumNodesPerCell; tNode++)
        {
            for(Plato::OrdinalType tDim = 0; tDim < PhysicsT::mNumSpatialDims; tDim++)
            {
                TEST_FLOATING_EQUALITY(tHostGoldCoords(tCell, tNode, tDim), tHostCoords(tCell, tNode, tDim), tTol);
                TEST_FLOATING_EQUALITY(tHostGoldVecOrdinals(tCell, tNode, tDim), tHostVectorFieldOrdinals(tCell, tNode, tDim), tTol);
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
    auto tHostScalarFieldOrdinals = Kokkos::create_mirror(tScalarFieldOrdinals);
    Kokkos::deep_copy(tHostScalarFieldOrdinals, tScalarFieldOrdinals);

    for(Plato::OrdinalType tNode = 0; tNode < PhysicsT::mNumNodesPerCell; tNode++)
    {
        for(Plato::OrdinalType tDim = 0; tDim < PhysicsT::mNumSpatialDims; tDim++)
        {
            TEST_FLOATING_EQUALITY(tHostGoldControlOrdinals(tNode, tDim), tHostControlOrdinals(tNode, tDim), tTol);
            TEST_FLOATING_EQUALITY(tHostGoldScalarOrdinals(tNode, tDim), tHostScalarFieldOrdinals(tNode, tDim), tTol);
        }
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
