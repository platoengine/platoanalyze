/*
 * ComputationalFluidDynamicsTests.cpp
 *
 *  Created on: Oct 13, 2020
 */

#include <Teuchos_UnitTestHarness.hpp>
#include <Teuchos_XMLParameterListCoreHelpers.hpp>

#include <unordered_map>

#include <Omega_h_assoc.hpp>
#include <Omega_h_shape.hpp>

#include "BLAS1.hpp"
#include "BLAS2.hpp"
#include "BLAS3.hpp"
#include "Simplex.hpp"
#include "Assembly.hpp"
#include "WorkSets.hpp"
#include "Variables.hpp"
#include "Solutions.hpp"
#include "NaturalBCs.hpp"
#include "UtilsOmegaH.hpp"
#include "Plato_Solve.hpp"
#include "UtilsTeuchos.hpp"
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

#include "Plato_Diagnostics.hpp"
#include "alg/PlatoSolverFactory.hpp"
#include "hyperbolic/SimplexFluids.hpp"
#include "hyperbolic/FluidsWorkSetsUtils.hpp"
#include "hyperbolic/SimplexFluidsFadTypes.hpp"
#include "hyperbolic/FluidsWorkSetBuilders.hpp"

#include "PlatoTestHelpers.hpp"

namespace Plato
{

namespace filesystem
{

/******************************************************************************//**
 * \fn exist
 *
 * \brief Return true if path exist; else, return false
 * \param [in] aPath directory/file path
 * \return boolean (true or false)
**********************************************************************************/
bool exist(const std::string &aPath)
{
    struct stat tBuf;
    int tReturn = stat(aPath.c_str(), &tBuf);
    return (tReturn == 0 ? true : false);
}
// function exist

/******************************************************************************//**
 * \fn exist
 *
 * \brief Delete file/directory if it exist
 * \param [in] aPath directory/file path
**********************************************************************************/
void remove(const std::string &aPath)
{
    if(Plato::filesystem::exist(aPath))
    {
        auto tCommand = std::string("rm -rf ") + aPath;
        std::system(tCommand.c_str());
    }
}
// function remove

}
// namespace filesystem


/******************************************************************************//**
 * \tparam SpaceDims (integer) number of spatial dimensions
 * \tparam NumNodes  (integer) number of nodes on surface/face
 * \tparam InStateType  input state type
 * \tparam OutStateType output state type
 *
 * \fn device_type inline project_vector_field_onto_surface
 *
 * \brief Project vector field onto surface/face
 *
 * \param [in] aCellOrdinal       cell/element ordinal
 * \param [in] aBasisFunctions    basis functions
 * \param [in] aLocalNodeOrdinals local cell node ordinals
 * \param [in] aInputState        input state
 * \param [in/out] aInputState    output state
**********************************************************************************/
template<Plato::OrdinalType SpaceDims,
         Plato::OrdinalType NumNodes,
         typename InStateType,
         typename OutStateType>
DEVICE_TYPE inline void
project_vector_field_onto_surface
(const Plato::OrdinalType & aCellOrdinal,
 const Plato::ScalarVector & aBasisFunctions,
 const Plato::OrdinalType aLocalNodeOrdinals[NumNodes],
 const Plato::ScalarMultiVectorT<InStateType> & aInputState,
 const Plato::ScalarMultiVectorT<OutStateType> & aOutputState)
{
    for(Plato::OrdinalType tDim = 0; tDim < SpaceDims; tDim++)
    {
        aOutputState(aCellOrdinal, tDim) = 0.0;
        for(Plato::OrdinalType tNode = 0; tNode < NumNodes; tNode++)
        {
            auto tLocalCellNode = aLocalNodeOrdinals[tNode];
            auto tLocalCellDof = (SpaceDims * tLocalCellNode) + tDim;
            aOutputState(aCellOrdinal, tDim) +=
                aBasisFunctions(tNode) * aInputState(aCellOrdinal, tLocalCellDof);
        }
    }
}
// function project_vector_field_onto_surface

/******************************************************************************//**
 * \tparam NumNodes  (integer) number of nodes on surface/face
 * \tparam InStateType  input state type
 * \tparam OutStateType output state type
 *
 * \fn device_type inline project_scalar_field_onto_surface
 *
 * \brief Project scalar field onto surface/face
 *
 * \param [in] aCellOrdinal       cell/element ordinal
 * \param [in] aBasisFunctions    basis functions
 * \param [in] aLocalNodeOrdinals local cell node ordinals
 * \param [in] aInputState        input state
 * \param [in/out] aInputState    output state
**********************************************************************************/
template<Plato::OrdinalType NumNodes,
         typename InStateType,
         typename OutStateType>
DEVICE_TYPE inline void
project_scalar_field_onto_surface
(const Plato::OrdinalType & aCellOrdinal,
 const Plato::ScalarVector & aBasisFunctions,
 const Plato::OrdinalType aLocalNodeOrdinals[NumNodes],
 const Plato::ScalarMultiVectorT<InStateType> & aInputState,
 const Plato::ScalarMultiVectorT<OutStateType> & aOutputState)
{
    aOutputState(aCellOrdinal) = 0.0;
    for(Plato::OrdinalType tNode = 0; tNode < NumNodes; tNode++)
    {
        auto tLocalCellNode = aLocalNodeOrdinals[tNode];
        aOutputState(aCellOrdinal) += aBasisFunctions(tNode) * aInputState(aCellOrdinal, tLocalCellNode);
    }
}
// function project_scalar_field_onto_surface







namespace Fluids
{




/***************************************************************************//**
 * \tparam PhysicsT    physics type
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
 * \brief Class responsible for the evaluation of the average surface pressure
 *   along the user-specified entity sets (e.g. side sets).
 *
 *                  \f[ \int_{\Gamma_e} p^n d\Gamma_e \f],
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
    CubatureRule mSurfaceCubatureRule; /*!< cubature integration rule on surface */
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
         mSurfaceCubatureRule(CubatureRule()),
         mSpatialDomain(aDomain),
         mFuncName(aName)
    {
        auto tMyCriteria = aInputs.sublist("Criteria").sublist(aName);
        mSideSets = Plato::teuchos::parse_array<std::string>("Sides", tMyCriteria);
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
        Plato::CreateFaceLocalNode2ElemLocalNodeIndexMap<mNumSpatialDims> tCreateFaceLocalNode2ElemLocalNodeIndexMap;

        // set input worksets
        auto tConfigWS = Plato::metadata<Plato::ScalarArray3DT<ConfigT>>(aWorkSets.get("configuration"));
        auto tCurrentPressWS = Plato::metadata<Plato::ScalarMultiVectorT<PressureT>>(aWorkSets.get("current pressure"));

        // transfer member data to device
        auto tCubatureWeight = mSurfaceCubatureRule.getCubWeight();
        auto tBasisFunctions = mSurfaceCubatureRule.getBasisFunctions();
        for(auto& tName : mSideSets)
        {
            // get faces on this side set
            auto tFaceOrdinalsOnSideSet = Plato::side_set_face_ordinals(mSpatialDomain.MeshSets, tName);
            auto tNumFaces = tFaceOrdinalsOnSideSet.size();
            Plato::ScalarArray3DT<ConfigT> tJacobians("face Jacobians", tNumFaces, mNumSpatialDimsOnFace, mNumSpatialDims);

            // set local worksets
            auto tNumCells = mSpatialDomain.Mesh.nelems();
            Plato::ScalarVectorT<PressureT> tCurrentPressGP("current pressure at Gauss point", tNumCells);

            Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumFaces), LAMBDA_EXPRESSION(const Plato::OrdinalType & aFaceI)
            {
                auto tFaceOrdinal = tFaceOrdinalsOnSideSet[aFaceI];
                // for all elements connected to this face, which is either 1 or 2 elements
                for( Plato::OrdinalType tElem = tFace2Elems_map[tFaceOrdinal]; tElem < tFace2Elems_map[tFaceOrdinal+1]; tElem++ )
                {
                    // create a map from face local node index to elem local node index
                    auto tCellOrdinal = tFace2Elems_elems[tElem];
                    Plato::OrdinalType tLocalNodeOrdinals[mNumSpatialDims];
                    tCreateFaceLocalNode2ElemLocalNodeIndexMap(tCellOrdinal, tFaceOrdinal, tCell2Verts, tFace2Verts, tLocalNodeOrdinals);

                    // calculate surface Jacobian and surface integral weight
                    ConfigT tSurfaceAreaTimesCubWeight(0.0);
                    tCalculateSurfaceJacobians(tCellOrdinal, aFaceI, tLocalNodeOrdinals, tConfigWS, tJacobians);
                    tCalculateSurfaceArea(aFaceI, tCubatureWeight, tJacobians, tSurfaceAreaTimesCubWeight);

                    // project current pressure onto surface
                    for(Plato::OrdinalType tNode = 0; tNode < mNumNodesPerFace; tNode++)
                    {
                        auto tLocalCellNode = tLocalNodeOrdinals[tNode];
                        tCurrentPressGP(tCellOrdinal) += tBasisFunctions(tNode) * tCurrentPressWS(tCellOrdinal, tLocalCellNode);
                    }

                    // calculate surface integral, which is defined as \int_{\Gamma_e}N_p^a p^h d\Gamma_e
                    for( Plato::OrdinalType tNode=0; tNode < mNumNodesPerFace; tNode++)
                    {
                        aResult(tCellOrdinal) += tBasisFunctions(tNode) * tCurrentPressGP(tCellOrdinal) * tSurfaceAreaTimesCubWeight;
                    }
                }
            }, "average surface pressure");

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
 * \brief Class responsible for the evaluation of the average surface temperature
 *   along the user-specified entity sets (e.g. side sets).
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
    static constexpr auto mNumTempDofsPerNode   = PhysicsT::SimplexT::mNumEnergyDofsPerNode;   /*!< number of temperature dofs per node */

    using ResultT = typename EvaluationT::ResultScalarType; /*!< result FAD type */
    using ConfigT = typename EvaluationT::ConfigScalarType; /*!< configuration FAD type */
    using CurrentTempT = typename EvaluationT::CurrentEnergyScalarType; /*!< current temperature FAD type */

    // set local typenames
    using CubatureRule  = Plato::LinearTetCubRuleDegreeOne<mNumSpatialDimsOnFace>; /*!< local short name for cubature rule class */

    // member metadata
    Plato::DataMap& mDataMap; /*!< holds output metadata */
    CubatureRule mSurfaceCubatureRule; /*!< cubature integration rule */
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
         mSurfaceCubatureRule(CubatureRule()),
         mSpatialDomain(aDomain),
         mFuncName(aName)
    {
        auto tMyCriteria = aInputs.sublist("Criteria").sublist(aName);
        mWallSets = Plato::teuchos::parse_array<std::string>("Sides", tMyCriteria);
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
    void evaluate
    (const Plato::WorkSets & aWorkSets,
     Plato::ScalarVectorT<ResultT> & aResult)
    const override
    { return; }

    /***************************************************************************//**
     * \fn void evaluateBoundary
     * \brief Evaluate scalar function along the computational boudary \f$ d\Gamma \f$.
     * \param [in] aWorkSets holds state work sets initialize with correct FAD types
     * \param [in] aResult   1D output work set of size number of cells
     ******************************************************************************/
    void evaluateBoundary
    (const Plato::WorkSets & aWorkSets,
     Plato::ScalarVectorT<ResultT> & aResult)
    const override
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
        Plato::CreateFaceLocalNode2ElemLocalNodeIndexMap<mNumSpatialDims> tCreateFaceLocalNode2ElemLocalNodeIndexMap;

        // set local data
        auto tCubatureWeight = mSurfaceCubatureRule.getCubWeight();
        auto tBasisFunctions = mSurfaceCubatureRule.getBasisFunctions();

        // set input worksets
        auto tConfigWS = Plato::metadata<Plato::ScalarArray3DT<ConfigT>>(aWorkSets.get("configuration"));
        auto tCurrentTempWS = Plato::metadata<Plato::ScalarMultiVectorT<CurrentTempT>>(aWorkSets.get("current temperature"));

        for(auto& tName : mWallSets)
        {
            // get faces on this side set
            auto tFaceOrdinalsOnSideSet = Plato::side_set_face_ordinals(mSpatialDomain.MeshSets, tName);
            auto tNumFaces = tFaceOrdinalsOnSideSet.size();
            Plato::ScalarArray3DT<ConfigT> tJacobians("face Jacobians", tNumFaces, mNumSpatialDimsOnFace, mNumSpatialDims);

            // set local worksets
            auto tNumCells = mSpatialDomain.Mesh.nelems();
            Plato::ScalarVectorT<CurrentTempT> tCurrentTempGP("current temperature at GP", tNumCells);

            Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumFaces), LAMBDA_EXPRESSION(const Plato::OrdinalType & aFaceI)
            {
                auto tFaceOrdinal = tFaceOrdinalsOnSideSet[aFaceI];
                // for all elements connected to this face, which is either 1 or 2 elements
                for( Plato::OrdinalType tElem = tFace2Elems_map[tFaceOrdinal]; tElem < tFace2Elems_map[tFaceOrdinal+1]; tElem++ )
                {
                    // create a map from face local node index to elem local node index
                    auto tCellOrdinal = tFace2Elems_elems[tElem];
                    Plato::OrdinalType tLocalNodeOrdinals[mNumSpatialDims];
                    tCreateFaceLocalNode2ElemLocalNodeIndexMap(tCellOrdinal, tFaceOrdinal, tCell2Verts, tFace2Verts, tLocalNodeOrdinals);

                    // calculate surface Jacobian and surface integral weight
                    ConfigT tSurfaceAreaTimesCubWeight(0.0);
                    tCalculateSurfaceJacobians(tCellOrdinal, aFaceI, tLocalNodeOrdinals, tConfigWS, tJacobians);
                    tCalculateSurfaceArea(aFaceI, tCubatureWeight, tJacobians, tSurfaceAreaTimesCubWeight);

                    // project current temperature onto surface
                    for(Plato::OrdinalType tNode = 0; tNode < mNumNodesPerFace; tNode++)
                    {
                        auto tLocalCellNode = tLocalNodeOrdinals[tNode];
                        tCurrentTempGP(tCellOrdinal) += tBasisFunctions(tNode) * tCurrentTempWS(tCellOrdinal, tLocalCellNode);
                    }

                    // calculate surface integral, which is defined as \int_{\Gamma_e}N_p^a T^h d\Gamma_e
                    for( Plato::OrdinalType tNode=0; tNode < mNumNodesPerFace; tNode++)
                    {
                        aResult(tCellOrdinal) += tBasisFunctions(tNode) * tCurrentTempGP(tCellOrdinal) * tSurfaceAreaTimesCubWeight;
                    }
                }
            }, "average surface temperature");

        }
    }
};
// class AverageSurfaceTemperature


/***************************************************************************//**
 * \tparam NumNodesPerCell number of nodes per cell (integer)
 * \tparam ControlT        control work set Forward Automatic Differentiation (FAD) type
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
template
<Plato::OrdinalType NumNodesPerCell,
typename ControlT>
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
template
<Plato::OrdinalType NumNodesPerCell,
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
 * \fn inline bool is_impermeability_defined
 *
 * \brief Return true if dimensionless impermeability number is defined; return
 *   false if it is not defined.
 *
 * \param [in] aInputs input file metadata
 *
 * \return boolean (true or false)
 ******************************************************************************/
inline bool
is_impermeability_defined
(Teuchos::ParameterList & aInputs)
{
    auto tHyperbolic = aInputs.sublist("Hyperbolic");
    if( !tHyperbolic.isSublist("Dimensionless Properties") )
    {
        THROWERR("Parameter Sublist 'Dimensionless Properties' is not defined.")
    }
    auto tSublist = tHyperbolic.sublist("Dimensionless Properties");
    return (tSublist.isParameter("Impermeability Number"));
}
// function is_impermeability_defined



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
/*
template<typename PhysicsT, typename EvaluationT>
class InternalDissipationEnergy : public Plato::Fluids::AbstractScalarFunction<PhysicsT, EvaluationT>
{
private:
    static constexpr auto mNumSpatialDims    = PhysicsT::SimplexT::mNumSpatialDims;         
    static constexpr auto mNumNodesPerCell   = PhysicsT::SimplexT::mNumNodesPerCell;        
    static constexpr auto mNumVelDofsPerNode = PhysicsT::SimplexT::mNumMomentumDofsPerCell; 

    // local forward automatic differentiation typenames
    using ResultT  = typename EvaluationT::ResultScalarType;          
    using CurVelT  = typename EvaluationT::CurrentMomentumScalarType;  
    using ConfigT  = typename EvaluationT::ConfigScalarType;           
    using ControlT = typename EvaluationT::ControlScalarType;          
    using StrainT  = typename Plato::Fluids::fad_type_t<typename PhysicsT::SimplexT, CurVelT, ConfigT>; 

    // set local typenames
    using CubatureRule  = Plato::LinearTetCubRuleDegreeOne<mNumSpatialDims>; 

    // member parameters
    std::string mFuncName;
    Plato::Scalar mImpermeability = 1.0; 
    Plato::Scalar mBrinkmanConvexityParam = 0.5; 

    // member metadata
    Plato::DataMap& mDataMap;
    CubatureRule mCubatureRule; 
    const Plato::SpatialDomain& mSpatialDomain; 

public:
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
        this->setImpermeability(aInputs);
        this->setBrinkmannModel(aInputs);
    }

    virtual ~InternalDissipationEnergy(){}

    std::string name() const override { return mFuncName; }

    void evaluate(const Plato::WorkSets & aWorkSets, Plato::ScalarVectorT<ResultT> & aResult) const override
    {
        // set local functors
        Plato::ComputeGradientWorkset<mNumSpatialDims> tComputeGradient;
        Plato::InterpolateFromNodal<mNumSpatialDims, mNumVelDofsPerNode, 0, mNumSpatialDims> tIntrplVectorField;

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
        auto tImpermeability = mImpermeability;
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
            Plato::blas3::scale<mNumSpatialDims, mNumSpatialDims>(aCellOrdinal, tTwoTimesPrNum, tStrainRate, tDevStress);
            Plato::blas3::dot<mNumSpatialDims, mNumSpatialDims>(aCellOrdinal, tDevStress, tDevStress, aResult);

            // calculate fictitious material model (i.e. brinkman model) contribution to internal energy
            ControlT tPenalizedPermeability = Plato::Fluids::brinkman_penalization<mNumNodesPerCell>
                (aCellOrdinal, tImpermeability, tBrinkConvexParam, tControlWS);
            tIntrplVectorField(aCellOrdinal, tBasisFunctions, tCurVelWS, tCurVelGP);
            Plato::blas2::dot<mNumSpatialDims>(aCellOrdinal, tCurVelGP, tCurVelGP, tCurVelDotCurVel);
            aResult(aCellOrdinal) += tPenalizedPermeability * tCurVelDotCurVel(aCellOrdinal);

            // apply gauss weight times volume multiplier
            aResult(aCellOrdinal) *= tVolumeTimesWeight(aCellOrdinal);

        }, "internal energy");
    }

    void evaluateBoundary(const Plato::WorkSets & aWorkSets, Plato::ScalarVectorT<ResultT> & aResult) const override
    { return; }

private:
    void setBrinkmannModel(Teuchos::ParameterList & aInputs)
    {
        auto tMyCriterionInputs = aInputs.sublist("Criteria").sublist(mFuncName);
        if(tMyCriterionInputs.isSublist("Penalty Function"))
        {
            auto tPenaltyFuncInputs = tMyCriterionInputs.sublist("Penalty Function");
            mBrinkmanConvexityParam = tPenaltyFuncInputs.get<Plato::Scalar>("Brinkman Convexity Parameter", 0.5);
        }
    }

    void setImpermeability
    (Teuchos::ParameterList & aInputs)
    {
        if(Plato::Fluids::is_impermeability_defined(aInputs))
        {
            auto tHyperbolic = aInputs.sublist("Hyperbolic");
            mImpermeability = Plato::teuchos::parse_parameter<Plato::Scalar>("Impermeability Number", "Dimensionless Properties", tHyperbolic);
        }
        else
        {
            auto tHyperbolic = aInputs.sublist("Hyperbolic");
            auto tDaNum = Plato::teuchos::parse_parameter<Plato::Scalar>("Darcy Number", "Dimensionless Properties", tHyperbolic);
            auto tPrNum = Plato::teuchos::parse_parameter<Plato::Scalar>("Prandtl Number", "Dimensionless Properties", tHyperbolic);
            mImpermeability = tPrNum / tDaNum;
        }
    }
};
*/
// class InternalDissipationEnergy


/***************************************************************************//**
 * \class CriterionBase
 *
 * This pure virtual class provides the template for a scalar functions of the form:
 *
 *    \f[ J = J(\phi, U^k, P^k, T^k, X) \f]
 *
 * Derived class are responsible for the evaluation of the function and its
 * corresponding derivatives with respect to control \f$\phi\f$, momentum
 * state \f$ U^k \f$, mass state \f$ P^k \f$, energy state \f$ T^k \f$, and
 * configuration \f$ X \f$ variables.
 ******************************************************************************/
class CriterionBase
{
public:
    virtual ~CriterionBase(){}

    /***************************************************************************//**
     * \fn std::string name
     * \brief Return scalar function name.
     * \return scalar function name
     ******************************************************************************/
    virtual std::string name() const = 0;

    /***************************************************************************//**
     * \fn Plato::Scalar value
     * \brief Return scalar function value.
     * \param [in] aControls control variables workset
     * \param [in] aPrimal   primal state database
     * \return scalar function value
     ******************************************************************************/
    virtual Plato::Scalar value
    (const Plato::ScalarVector & aControls,
     const Plato::Primal & aPrimal) const = 0;

    /***************************************************************************//**
     * \fn Plato::ScalarVector gradientConfig
     * \brief Return scalar function derivative with respect to the configuration variables.
     *
     * \param [in] aControls control variables workset
     * \param [in] aPrimal   primal state database
     *
     * \return scalar function derivative with respect to the configuration variables
     ******************************************************************************/
    virtual Plato::ScalarVector gradientConfig
    (const Plato::ScalarVector & aControls,
     const Plato::Primal & aPrimal) const = 0;

    /***************************************************************************//**
     * \fn Plato::ScalarVector gradientControl
     * \brief Return scalar function derivative with respect to the control variables.
     *
     * \param [in] aControls control variables workset
     * \param [in] aPrimal   primal state database
     *
     * \return scalar function derivative with respect to the control variables
     ******************************************************************************/
    virtual Plato::ScalarVector gradientControl
    (const Plato::ScalarVector & aControls,
     const Plato::Primal & aPrimal) const = 0;

    /***************************************************************************//**
     * \fn Plato::ScalarVector gradientCurrentPress
     * \brief Return scalar function derivative with respect to the current pressure.
     *
     * \param [in] aControls control variables workset
     * \param [in] aPrimal   primal state database
     *
     * \return scalar function derivative with respect to the current pressure
     ******************************************************************************/
    virtual Plato::ScalarVector gradientCurrentPress
    (const Plato::ScalarVector & aControls,
     const Plato::Primal & aPrimal) const = 0;

    /***************************************************************************//**
     * \fn Plato::ScalarVector gradientCurrentTemp
     * \brief Return scalar function derivative with respect to the current temperature.
     *
     * \param [in] aControls control variables workset
     * \param [in] aPrimal   primal state database
     *
     * \return scalar function derivative with respect to the current temperature
     ******************************************************************************/
    virtual Plato::ScalarVector gradientCurrentTemp
    (const Plato::ScalarVector & aControls,
     const Plato::Primal & aPrimal) const = 0;

    /***************************************************************************//**
     * \fn Plato::ScalarVector gradientCurrentVel
     * \brief Return scalar function derivative with respect to the current velocity.
     *
     * \param [in] aControls control variables workset
     * \param [in] aPrimal   primal state database
     *
     * \return scalar function derivative with respect to the current velocity
     ******************************************************************************/
    virtual Plato::ScalarVector gradientCurrentVel
    (const Plato::ScalarVector & aControls,
     const Plato::Primal & aPrimal) const = 0;
};
// class CriterionBase


/***************************************************************************//**
 * \tparam PhysicsT fluid flow physics type
 *
 * \class ScalarFunction
 *
 * Class manages the evaluation of a scalar functions in the form:
 *
 *                  \f[ J(\phi, U^k, P^k, T^k, X) \f]
 *
 * Responsabilities include evaluation of the partial derivatives with respect
 * to control \f$\phi\f$, momentum state \f$ U^k \f$, mass state \f$ P^k \f$,
 * energy state \f$ T^k \f$ and configuration \f$ X \f$.
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
    using ValueFunc        = std::shared_ptr<Plato::Fluids::AbstractScalarFunction<PhysicsT, ResidualEvalT>>;     /*!< short name/notation for a scalar function of residual FAD evaluation type */
    using GradConfigFunc   = std::shared_ptr<Plato::Fluids::AbstractScalarFunction<PhysicsT, GradConfigEvalT>>;   /*!< short name/notation for a scalar function of partial wrt configuration FAD evaluation type */
    using GradControlFunc  = std::shared_ptr<Plato::Fluids::AbstractScalarFunction<PhysicsT, GradControlEvalT>>;  /*!< short name/notation for a scalar function of partial wrt control FAD evaluation type */
    using GradCurVelFunc   = std::shared_ptr<Plato::Fluids::AbstractScalarFunction<PhysicsT, GradCurVelEvalT>>;   /*!< short name/notation for a scalar function of partial wrt current velocity state FAD evaluation type */
    using GradCurTempFunc  = std::shared_ptr<Plato::Fluids::AbstractScalarFunction<PhysicsT, GradCurTempEvalT>>;  /*!< short name/notation for a scalar function of partial wrt current temperature state FAD evaluation type */
    using GradCurPressFunc = std::shared_ptr<Plato::Fluids::AbstractScalarFunction<PhysicsT, GradCurPressEvalT>>; /*!< short name/notation for a scalar function of partial wrt current pressure state FAD evaluation type */

    // element scalar functions per element block, i.e. domain
    std::unordered_map<std::string, ValueFunc>        mValueFuncs; /*!< map from domain (i.e. element block) to scalar function of residual FAD evaluation type */
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
            mValueFuncs.at(tName)->evaluate(tInputWorkSets, tResultWS);

            tReturnValue += Plato::local_result_sum<Plato::Scalar>(tNumCells, tResultWS);
        }

        // evaluate boundary
        {
            Plato::WorkSets tInputWorkSets;
            auto tNumCells = mSpatialModel.Mesh.nelems();
            Plato::Fluids::build_scalar_function_worksets<ResidualEvalT>
                (tNumCells, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            Plato::ScalarVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells);
            mValueFuncs.begin()->second->evaluateBoundary(tInputWorkSets, tResultWS);

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

            mValueFuncs[tName] =
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
 * \tparam PhysicsT    Physics type
 * \tparam EvaluationT Forward Automatic Differentiation (FAD) evaluation type
 *
 * \class AbstractVectorFunction
 *
 * \brief Pure virtual base class for vector functions.
 ******************************************************************************/
template<typename PhysicsT, typename EvaluationT>
class AbstractVectorFunction
{
private:
    using ResultT = typename EvaluationT::ResultScalarType;

public:
    AbstractVectorFunction(){}
    virtual ~AbstractVectorFunction(){}

    /***************************************************************************//**
     * \fn void evaluate
     * \brief Evaluate vector function within the domain.
     *
     * \param [in] aWorkSets holds state and control worksets
     * \param [in/out] aResult   result workset
     ******************************************************************************/
    virtual void evaluate
    (const Plato::WorkSets & aWorkSets,
     Plato::ScalarMultiVectorT<ResultT> & aResult) const = 0;

    /***************************************************************************//**
     * \fn void evaluateBoundary
     * \brief Evaluate vector function on non-prescribed boundaries.
     *
     * \param [in] aWorkSets holds state and control worksets
     * \param [in/out] aResult   result workset
     ******************************************************************************/
    virtual void evaluateBoundary
    (const Plato::SpatialModel & aSpatialModel,
     const Plato::WorkSets & aWorkSets,
     Plato::ScalarMultiVectorT<ResultT> & aResult) const = 0;

    /***************************************************************************//**
     * \fn void evaluatePrescribed
     * \brief Evaluate vector function on prescribed boundaries.
     *
     * \param [in] aWorkSets holds state and control worksets
     * \param [in/out] aResult   result workset
     ******************************************************************************/
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
 * \f[ \alpha\bar{u}_j^n \frac{\partial \bar{u}_i^n}{\partial\bar{x}_j} \f]
 *
 * where \f$\alpha\f$ is a scalar multiplier, \f$ u_i \f$ is the i-th velocity
 * component and \f$ x_i \f$ is the i-th coordinate.
 *
 * \param [in] aCellOrdinal cell/element ordinal
 * \param [in] aGradient    spatial gradient workset
 * \param [in] aPrevVelWS   previous velocity workset
 * \param [in] aPrevVelGP   previous velocity evaluated at Gauss points
 * \param [in] aMultiplier  scalar multiplier (default = 1.0)
 * \param [in/out] aResult  result/output workset
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
 * \brief Integrate element viscous forces, defined as
 *
 * \f[ \alpha\int_{\Omega}\frac{\partial w_i^h}{\partial\bar{x}_j}\bar\tau_{ij}^n\,d\Omega \f]
 *
 * where \f$\bar\tau_{ij}^n\f$ is the deviatoric stress tensor and \f$\alpha\f$
 * is a scalar multiplier.
 *
 * \param [in] aCellOrdinal   cell/element ordinal
 * \param [in] aPrandtlNumber dimensionless Prandtl number
 * \param [in] aCellVolume    cell/element volume workset
 * \param [in] aGradient      spatial gradient workset
 * \param [in] aStrainRate    strain workset
 * \param [in] aMultiplier    scalar multiplier (default = 1.0)
 * \param [in/out] aResult    result/output workset
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
 * \brief Calculate natural convective forces, defined as
 *
 * \f[ \alpha Gr_i Pr^2\bar{T}^n \f]
 *
 * where \f$\alpha\f$ denotes a scalar multiplier, \f$ Gr_i \f$ is the Grashof
 * number, \f$ Pr \f$ is the Prandtl number and \f$ T^n \f$ is the temperature
 * field at time step n.
 *
 * \param [in] aCellOrdinal   cell/element ordinal
 * \param [in] aPrTimesPr     Prandtl number squared
 * \param [in] aGrashofNum    Grashof number
 * \param [in] aPrevTempGP    previous temperature at Gauss points
 * \param [in] aMultiplier    scalar multiplier (default = 1.0)
 * \param [in/out] aResult    result/output workset
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
 * \f[ \alpha\beta u^{n}_i \f]
 *
 * where \f$\alpha\f$ is a scalar multiplier, \f$ \beta \f$ is the dimensionless
 * impermeability constant and \f$u_i^{n}\f$ is i-th component of the velocity
 * field at time step n.
 *
 * \param [in] aCellOrdinal    cell/element ordinal
 * \param [in] aImpermeability impermeability constant
 * \param [in] aPrevTempGP     previous velocity at Gauss points
 * \param [in] aMultiplier     scalar multiplier (default = 1.0)
 * \param [in/out] aResult     result/output workset
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
 const ControlT & aImpermeability,
 const Plato::ScalarMultiVectorT<PrevVelT> & aPrevVelGP,
 const Plato::ScalarMultiVectorT<ResultT> & aResult,
 Plato::Scalar aMultiplier = 1.0)
{
    for (Plato::OrdinalType tDim = 0; tDim < SpaceDim; tDim++)
    {
        aResult(aCellOrdinal, tDim) += aMultiplier * aImpermeability * aPrevVelGP(aCellOrdinal, tDim);
    }
}
// function calculate_brinkman_forces


/***************************************************************************//**
 * \tparam NumNodes number of nodes in cell/element (integer)
 * \tparam SpaceDim   spatial dimensions (integer)
 * \tparam ResultT    output Forward Automatic Differentiation (FAD) type
 * \tparam ConfigT    configuration FAD type
 * \tparam PrevVelT   previous velocity FAD type
 * \tparam StabilityT stabilizing force FAD type
 *
 * \fn device_type inline void integrate_stabilizing_vector_force
 *
 * \brief Integrate stabilizing momentum forces, defined as
 *
 * \f[
 *   \alpha\int_{\Omega} \left( \frac{\partial w_i^h}{\partial\bar{x}_k}\bar{u}^n_k \right) \hat{S}^n_{\bar{u}_i}\, d\Omega
 * \f]
 *
 * where \f$\alpha\f$ denotes a scalar multiplier, \f$ u_k^n \f$ is the k-th
 * component of the velocity field at time step n, \f$ x_i \f$ is the i-th
 * coordinate and \f$\hat{S}^n_{\bar{u}_i}\f$ is the stabilizing force.
 *
 * \param [in] aCellOrdinal    cell/element ordinal
 * \param [in] aCellVolume     cell/element volume workset
 * \param [in] aGradient       spatial gradient workset
 * \param [in] aPrevVelGP      previous velocity at Gauss points
 * \param [in] aStabilization  stabilization forces
 * \param [in] aMultiplier     scalar multiplier (default = 1.0)
 * \param [in/out] aResult     result/output workset
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
 * \f[ \alpha\int_{\Omega} w_i^h f_i d\Omega \f]
 *
 * where \f$\alpha\f$ denotes a scalar multiplier.
 *
 * \param [in] aCellOrdinal    cell/element ordinal
 * \param [in] aBasisFunctions cell/element basis functions
 * \param [in] aCellVolume     cell/element volume workset
 * \param [in] aGradient       spatial gradient workset
 * \param [in] aField          vector field
 * \param [in] aMultiplier     scalar multiplier (default = 1.0)
 * \param [in/out] aResult     result/output workset
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
 * \fn inline Plato::Scalar reynolds_number
 *
 * \brief Parse Reynolds number from input file.
 * \param [in] aInputs input file metadata
 * \return Reynolds number
 ******************************************************************************/
inline Plato::Scalar
reynolds_number
(Teuchos::ParameterList & aInputs)
{
    if(aInputs.isSublist("Hyperbolic") == false)
    {
        THROWERR("'Hyperbolic' Parameter List is not defined.")
    }
    auto tHyperbolic = aInputs.sublist("Hyperbolic");
    auto tReNum = Plato::teuchos::parse_parameter<Plato::Scalar>("Reynolds Number", "Dimensionless Properties", tHyperbolic);
    return tReNum;
}
// function reynolds_number

/***************************************************************************//**
 * \fn inline Plato::Scalar prandtl_number
 *
 * \brief Parse Prandtl number from input file.
 * \param [in] aInputs input file metadata
 * \return Prandtl number
 ******************************************************************************/
inline Plato::Scalar
prandtl_number
(Teuchos::ParameterList & aInputs)
{
    if(aInputs.isSublist("Hyperbolic") == false)
    {
        THROWERR("'Hyperbolic' Parameter List is not defined.")
    }
    auto tHyperbolic = aInputs.sublist("Hyperbolic");
    auto tPrNum = Plato::teuchos::parse_parameter<Plato::Scalar>("Prandtl Number", "Dimensionless Properties", tHyperbolic);
    return tPrNum;
}
// function prandtl_number

/***************************************************************************//**
 * \fn inline bool calculate_brinkman_forces
 *
 * \brief Return true if Brinkman forces are enabled, return false if disabled.
 * \param [in] aInputs input file metadata
 * \return boolean (true or false)
 ******************************************************************************/
inline bool calculate_brinkman_forces
(Teuchos::ParameterList& aInputs)
{
    if(aInputs.isSublist("Hyperbolic") == false)
    {
        THROWERR("'Hyperbolic' Parameter List is not defined.")
    }

    auto tHyperbolic = aInputs.sublist("Hyperbolic");
    auto tScenario = tHyperbolic.get<std::string>("Scenario", "Analysis");
    auto tLowerScenario = Plato::tolower(tScenario);
    if(tLowerScenario == "density to")
    {
	return true;
    }
    return false;
}
// function calculate_brinkman_forces

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
 * \brief Returns true if energy equation is enabled, else, returns false.
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
 * \fn inline bool calculate_effective_conductivity
 *
 * \brief Calculate effective conductivity based on the heat transfer mechanism requested.
 * \param [in] aInputs input file metadata
 * \return effective conductivity
 ******************************************************************************/
inline Plato::Scalar
calculate_effective_conductivity
(Teuchos::ParameterList & aInputs)
{
    auto tHyperbolic = aInputs.sublist("Hyperbolic");
    auto tTag = tHyperbolic.get<std::string>("Heat Transfer", "natural");
    auto tHeatTransfer = Plato::tolower(tTag);

    auto tOutput = 0;
    if(tHeatTransfer == "forced" || tHeatTransfer == "mixed")
    {
        auto tPrNum = Plato::teuchos::parse_parameter<Plato::Scalar>("Prandtl Number", "Dimensionless Properties", tHyperbolic);
        auto tReNum = Plato::teuchos::parse_parameter<Plato::Scalar>("Reynolds Number", "Dimensionless Properties", tHyperbolic);
        tOutput = static_cast<Plato::Scalar>(1) / (tReNum*tPrNum);
    }
    else if(tHeatTransfer == "natural")
    {
        tOutput = 1.0;
    }
    else
    {
        THROWERR(std::string("'Heat Transfer' mechanism with tag '") + tHeatTransfer + "' is not supported.")
    }
    return tOutput;
}
// function calculate_effective_conductivity

/***************************************************************************//**
 * \fn inline Plato::Scalar calculate_viscosity_constant
 *
 * \brief Calculate dimensionless viscocity \f$ \nu f\$ constant. The dimensionless
 * viscocity is given by \f$ \nu=\frac{1}{Re} f\$ if forced convection dominates or
 * by \f$ \nu=Pr \f$ is natural convection dominates.
 *
 * \param [in] aInputs input file metadata
 *
 * \return dimensionless viscocity
 ******************************************************************************/
inline Plato::Scalar
calculate_viscosity_constant
(Teuchos::ParameterList & aInputs)
{
    auto tHeatTransfer = Plato::Fluids::heat_transfer_tag(aInputs);
    if(tHeatTransfer == "forced" || tHeatTransfer == "mixed" || tHeatTransfer == "none")
    {
        auto tReNum = Plato::Fluids::reynolds_number(aInputs);
        auto tViscocity = static_cast<Plato::Scalar>(1) / tReNum;
        return tViscocity;
    }
    else if(tHeatTransfer == "natural")
    {
        auto tViscocity = Plato::Fluids::prandtl_number(aInputs);
        return tViscocity;
    }
    else
    {
        THROWERR(std::string("'Heat Transfer' mechanism with tag '") + tHeatTransfer + "' is not supported.")
    }
}
// function calculate_viscosity_constant


/***************************************************************************//**
 * \fn inline Plato::Scalar buoyancy_constant_natural_convection_problems
 *
 * \brief Calculate buoyancy constant for natural convection problems.
 *
 * \param [in] aInputs input file metadata
 *
 * \return dimensionless buoyancy constant
 ******************************************************************************/
inline Plato::Scalar
buoyancy_constant_natural_convection_problems
(Teuchos::ParameterList & aInputs)
{
    auto tPrNum = Plato::Fluids::prandtl_number(aInputs);
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
// function buoyancy_constant_natural_convection_problems


/***************************************************************************//**
 * \fn inline Plato::Scalar buoyancy_constant_mixed_convection_problems
 *
 * \brief Calculate buoyancy constant for mixed convection problems.
 *
 * \param [in] aInputs input file metadata
 *
 * \return dimensionless buoyancy constant
 ******************************************************************************/
inline Plato::Scalar
buoyancy_constant_mixed_convection_problems
(Teuchos::ParameterList & aInputs)
{
    if(Plato::Fluids::is_dimensionless_parameter_defined("Richardson Number", aInputs))
    {
        return static_cast<Plato::Scalar>(1.0);
    }
    else if(Plato::Fluids::is_dimensionless_parameter_defined("Grashof Number", aInputs))
    {
        auto tReNum = Plato::Fluids::reynolds_number(aInputs);
        auto tBuoyancy = static_cast<Plato::Scalar>(1.0) / (tReNum * tReNum);
        return tBuoyancy;
    }
    else
    {
        THROWERR("Mixed convection properties are not defined. One of these two options should be provided: 'Grashof Number' or 'Richardson Number'")
    }
}
// function buoyancy_constant_mixed_convection_problems

/***************************************************************************//**
 * \fn inline Plato::Scalar calculate_buoyancy_constant
 *
 * \brief Calculate dimensionless buoyancy constant \f$ \beta f\$. The buoyancy
 * constant is defined by \f$ \beta=\frac{1}{Re^2} f\$ if forced convection dominates.
 * In contrast, the buoyancy constant for natural convection dominated problems
 * is given by \f$ \nu=Pr^2 \f$ or \f$ \nu=Pr \f$ depending on which dimensionless
 * convective constant was provided by the user (e.g. Rayleigh or Grashof number).
 *
 * \param [in] aInputs input file metadata
 *
 * \return dimensionless buoyancy constant
 ******************************************************************************/
inline Plato::Scalar
calculate_buoyancy_constant
(Teuchos::ParameterList & aInputs)
{
    Plato::Scalar tBuoyancy = 0.0; // heat transfer calculations inactive if buoyancy = 0.0

    auto tHeatTransfer = Plato::Fluids::heat_transfer_tag(aInputs);
    if(tHeatTransfer == "mixed")
    {
        tBuoyancy = Plato::Fluids::buoyancy_constant_mixed_convection_problems(aInputs);
    }
    else if(tHeatTransfer == "natural")
    {
        tBuoyancy = Plato::Fluids::buoyancy_constant_natural_convection_problems(aInputs);
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
// function calculate_buoyancy_constant


/***************************************************************************//**
 * \tparam SpaceDim spatial dimensions (integer)
 *
 * \fn inline Plato::ScalarVector rayleigh_number
 *
 * \brief Parse dimensionless Rayleigh constants.
 *
 * \param [in] aInputs input file metadata
 *
 * \return Rayleigh constants
 ******************************************************************************/
template<Plato::OrdinalType SpaceDim>
inline Plato::ScalarVector
rayleigh_number
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
        auto tRaNum = Plato::teuchos::parse_parameter<Teuchos::Array<Plato::Scalar>>("Rayleigh Number", "Dimensionless Properties", tHyperbolic);
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
// function rayleigh_number

/***************************************************************************//**
 * \tparam SpaceDim spatial dimensions (integer)
 *
 * \fn inline Plato::ScalarVector grashof_number
 *
 * \brief Parse dimensionless Grashof constants.
 *
 * \param [in] aInputs input file metadata
 *
 * \return Grashof constants
 ******************************************************************************/
template<Plato::OrdinalType SpaceDim>
inline Plato::ScalarVector
grashof_number
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
        auto tGrNum = Plato::teuchos::parse_parameter<Teuchos::Array<Plato::Scalar>>("Grashof Number", "Dimensionless Properties", tHyperbolic);
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
// function grashof_number

/***************************************************************************//**
 * \tparam SpaceDim spatial dimensions (integer)
 *
 * \fn inline Plato::ScalarVector richardson_number
 *
 * \brief Parse dimensionless Richardson constants.
 *
 * \param [in] aInputs input file metadata
 *
 * \return Richardson constants
 ******************************************************************************/
template<Plato::OrdinalType SpaceDim>
inline Plato::ScalarVector
richardson_number
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
        auto tRiNum = Plato::teuchos::parse_parameter<Teuchos::Array<Plato::Scalar>>("Richardson Number", "Dimensionless Properties", tHyperbolic);
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
// function richardson_number


/***************************************************************************//**
 * \tparam SpaceDim spatial dimensions (integer)
 *
 * \fn inline Plato::ScalarVector parse_natural_convection_number
 *
 * \brief Parse dimensionless natural convection constants (e.g. Rayleigh or Grashof number).
 *
 * \param [in] aInputs input file metadata
 *
 * \return natural convection constants
 ******************************************************************************/
template<Plato::OrdinalType SpaceDim>
inline Plato::ScalarVector
parse_natural_convection_number
(Teuchos::ParameterList & aInputs)
{
    auto tHeatTransfer = Plato::Fluids::heat_transfer_tag(aInputs);
    if( Plato::Fluids::is_dimensionless_parameter_defined("Rayleigh Number", aInputs) &&
            (tHeatTransfer == "natural") )
    {
        return (Plato::Fluids::rayleigh_number<SpaceDim>(aInputs));
    }
    else if( Plato::Fluids::is_dimensionless_parameter_defined("Grashof Number", aInputs) &&
            (tHeatTransfer == "natural" || tHeatTransfer == "mixed") )
    {
        return (Plato::Fluids::grashof_number<SpaceDim>(aInputs));
    }
    else if( Plato::Fluids::is_dimensionless_parameter_defined("Richardson Number", aInputs) &&
            (tHeatTransfer == "mixed") )
    {
        return (Plato::Fluids::richardson_number<SpaceDim>(aInputs));
    }
    else
    {
        THROWERR(std::string("Natural convection properties are not defined. One of these options") +
                 " should be provided: 'Grashof Number' (for natural or mixed convection problems), " +
                 "'Rayleigh Number' (for natural convection problems), or 'Richardson Number' (for mixed convection problems).")
    }
}
// function parse_natural_convection_number

/***************************************************************************//**
 * \fn inline Plato::Scalar stabilization_constant
 *
 * \brief Parse stabilization force scalar multiplier.
 *
 * \param [in] aSublistName parameter sublist name
 * \param [in] aInputs      input file metadata
 *
 * \return scalar multiplier
 ******************************************************************************/
inline Plato::Scalar 
stabilization_constant
(const std::string & aSublistName,
 Teuchos::ParameterList & aInputs)
{
    if(aInputs.isSublist("Hyperbolic") == false)
    {
        THROWERR("'Hyperbolic' Parameter List is not defined.")
    }

    auto tOutput = 0.0;
    auto tHyperbolic = aInputs.sublist("Hyperbolic");
    if(tHyperbolic.isSublist(aSublistName))
    {
        auto tMomentumConservation = tHyperbolic.sublist(aSublistName);
        tOutput = tMomentumConservation.get<double>("Stabilization Constant", 0.0);
    }
    return tOutput;
}
// function stabilization_constant





/***************************************************************************//**
 * \tparam PhysicsT    Physics Type
 * \tparam EvaluationT Forward Automatic Differentiation (FAD) Evaluation Type
 *
 * \class ThermalBuoyancy
 *
 * \brief Class responsible for evaluating thermal buoyancy forces, including
 *   stabilization forces associated with the thermal buoyancy forces.
 *
 * Thermal Buoyancy Force:
 *   \[ \Delta{t}*Bu*Gr_i \int_{\Omega_e} w T^n d\Omega_e \]
 *
 * Stabilized Buoyancy Force:
 *   \[ \frac{\Delta{t}^2}{2}*Bu*Gr_i \int_{\Omega_e} (\frac{\partial w}{\partial x_k} u_k^n ) T^n d\Omega_e \]
 *
 ******************************************************************************/
template<typename PhysicsT, typename EvaluationT>
class ThermalBuoyancy
{
private:
    static constexpr auto mNumSpatialDims       = PhysicsT::SimplexT::mNumSpatialDims;         /*!< number of spatial dimensions */
    static constexpr auto mNumNodesPerCell      = PhysicsT::SimplexT::mNumNodesPerCell;        /*!< number of nodes per cell */
    static constexpr auto mNumVelDofsPerNode    = PhysicsT::SimplexT::mNumMomentumDofsPerNode; /*!< number of momentum dofs per node */
    static constexpr auto mNumTempDofsPerNode   = PhysicsT::SimplexT::mNumEnergyDofsPerNode;   /*!< number of energy dofs per node */

    // set local ad types
    using ResultT   = typename EvaluationT::ResultScalarType; /*!< result FAD type */
    using ConfigT   = typename EvaluationT::ConfigScalarType; /*!< configuration FAD type */
    using PrevVelT  = typename EvaluationT::PreviousMomentumScalarType; /*!< previous velocity FAD type */
    using PrevTempT = typename EvaluationT::PreviousEnergyScalarType; /*!< previous temperature FAD type */

    Plato::Scalar mStabilization = 0.0; /*!< stabilization constant */
    Plato::Scalar mBuoyancyConst = 0.0; /*!< dimensionless buoyancy constant */
    Plato::Scalar mBuoyancyDamping = 1.0; /*!< artificial buoyancy damping */
    Plato::ScalarVector mNaturalConvectionNum; /*!< dimensionless natural convection number (either Rayleigh or Grashof - depends on user's input) */

    Plato::DataMap& mDataMap; /*!< output database */
    const Plato::SpatialDomain& mSpatialDomain; /*!< Plato spatial model */
    Plato::LinearTetCubRuleDegreeOne<mNumSpatialDims> mCubatureRule; /*!< cubature rule evaluator */

public:
    /***************************************************************************//**
     * \brief Constructor
     * \param [in] aDomain  spatial domain metadata
     * \param [in] aDataMap output data map
     * \param [in] aInputs  input file metadata
     ******************************************************************************/
    ThermalBuoyancy
    (const Plato::SpatialDomain & aDomain,
     Plato::DataMap             & aDataMap,
     Teuchos::ParameterList     & aInputs) :
         mDataMap(aDataMap),
         mSpatialDomain(aDomain),
         mCubatureRule(Plato::LinearTetCubRuleDegreeOne<mNumSpatialDims>())
    {
        this->setAritificalDamping(aInputs);
        mBuoyancyConst = Plato::Fluids::calculate_buoyancy_constant(aInputs);
        mStabilization = Plato::Fluids::stabilization_constant("Momentum Conservation", aInputs);
        mNaturalConvectionNum = Plato::Fluids::parse_natural_convection_number<mNumSpatialDims>(aInputs);
    }

    /***************************************************************************//**
     * \brief Destructor
     ******************************************************************************/
    ~ThermalBuoyancy(){}

    /***************************************************************************//**
     * \brief Evaluate thermal buoyancy forces, including stabilized forces if enabled.
     * \param [in] aWorkSets holds input worksets (e.g. states, control, etc)
     * \param [in/out] aResultWS result/output workset
     ******************************************************************************/
    void evaluate
    (const Plato::WorkSets & aWorkSets,
     Plato::ScalarMultiVectorT<ResultT> & aResultWS) const
    {
        auto tNumCells = mSpatialDomain.numCells();
        if( tNumCells != static_cast<Plato::OrdinalType>(aResultWS.extent(0)) )
        {
            THROWERR(std::string("Number of elements mismatch. Spatial domain and output/result workset ")
                + "have different number of cells. " + "Spatial domain has '" + std::to_string(tNumCells)
                + "' elements and output workset has '" + std::to_string(aResultWS.extent(0)) + "' elements.")
        }

        // set local functors
        Plato::ComputeGradientWorkset<mNumSpatialDims> tComputeGradient;
        Plato::InterpolateFromNodal<mNumSpatialDims, mNumTempDofsPerNode> tIntrplScalarField;
        Plato::InterpolateFromNodal<mNumSpatialDims, mNumVelDofsPerNode, 0/*offset*/, mNumSpatialDims> tIntrplVectorField;

        // set temporary worksets
        Plato::ScalarVectorT<ConfigT> tCellVolume("cell weight", tNumCells);
        Plato::ScalarArray3DT<ConfigT> tGradient("cell gradient", tNumCells, mNumNodesPerCell, mNumSpatialDims);
        Plato::ScalarVectorT<PrevTempT> tPrevTempGP("previous temperature at Gauss point", tNumCells);
        Plato::ScalarMultiVectorT<PrevVelT> tPrevVelGP("previous velocity", tNumCells, mNumSpatialDims);
        Plato::ScalarMultiVectorT<ResultT> tThermalBuoyancy("thermal buoyancy", tNumCells, mNumSpatialDims);

        // set input worksets
        auto tConfigWS  = Plato::metadata<Plato::ScalarArray3DT<ConfigT>>(aWorkSets.get("configuration"));
        auto tPrevVelWS = Plato::metadata<Plato::ScalarMultiVectorT<PrevVelT>>(aWorkSets.get("previous velocity"));
        auto tPrevTempWS = Plato::metadata<Plato::ScalarMultiVectorT<PrevTempT>>(aWorkSets.get("previous temperature"));
        auto tCriticalTimeStep = Plato::metadata<Plato::ScalarVector>(aWorkSets.get("critical time step"));

        // transfer member data to device
        auto tStabilization = mStabilization;
        auto tBuoyancyConst = mBuoyancyConst;
        auto tBuoyancyDamping = mBuoyancyDamping;
        auto tNaturalConvectionNum = mNaturalConvectionNum;

        auto tCubWeight = mCubatureRule.getCubWeight();
        auto tBasisFunctions = mCubatureRule.getBasisFunctions();
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
        {
            tComputeGradient(aCellOrdinal, tGradient, tConfigWS, tCellVolume);
            tCellVolume(aCellOrdinal) *= tCubWeight;

            // 1. add previous buoyancy force to residual, i.e. R += (\Delta{t}*Bu*Gr_i) M T_n, where Bu is the buoyancy constant
            auto tMultiplier = tBuoyancyDamping * tCriticalTimeStep(0);
            tIntrplScalarField(aCellOrdinal, tBasisFunctions, tPrevTempWS, tPrevTempGP);
            Plato::Fluids::calculate_natural_convective_forces<mNumSpatialDims>
                (aCellOrdinal, tBuoyancyConst, tNaturalConvectionNum, tPrevTempGP, tThermalBuoyancy);
            Plato::Fluids::integrate_vector_field<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tBasisFunctions, tCellVolume, tThermalBuoyancy, aResultWS, -tMultiplier);

            // 2. add stabilizing buoyancy force to residual. i.e. R += \frac{\Delta{t}^2}{2} Bu*Gr_i) M T_n
            tIntrplVectorField(aCellOrdinal, tBasisFunctions, tPrevVelWS, tPrevVelGP);
            tMultiplier = tStabilization * static_cast<Plato::Scalar>(0.5) * tBuoyancyDamping * tCriticalTimeStep(0) * tCriticalTimeStep(0);
            Plato::Fluids::integrate_stabilizing_vector_force<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tCellVolume, tGradient, tPrevVelGP, tThermalBuoyancy, aResultWS, -tMultiplier);
        }, "add contribution from thermal buoyancy forces to residual");
    }

private:
    /***************************************************************************//**
     * \fn void setAritificalDamping
     * \brief Set artificial buoyancy damping parameter.
     * \param [in] aInputs  input file metadata
     ******************************************************************************/
    void setAritificalDamping(Teuchos::ParameterList& aInputs)
    {
        if(aInputs.isSublist("Hyperbolic") == false)
        {
            THROWERR("'Hyperbolic' Parameter List is not defined.")
        }
        auto tHyperbolic = aInputs.sublist("Hyperbolic");
        if(tHyperbolic.isSublist("Momentum Conservation"))
        {
            auto tMomentumConservation = tHyperbolic.sublist("Momentum Conservation");
            mBuoyancyDamping = tMomentumConservation.get<Plato::Scalar>("Buoyancy Damping", 1.0);
        }
    }
};
// class ThermalBuoyancy







/***************************************************************************//**
 * \tparam PhysicsT    Physics Type
 * \tparam EvaluationT Forward Automatic Differentiation (FAD) Evaluation Type
 *
 * \class BrinkmanForces
 *
 * \brief Class responsible for the evaluation of the Brinkman forces (i.e.
 *   fictitious material model), including the stabilization forces associated
 *   with the Brinkman forces.
 *
 * Thermal Buoyancy Force:
 *   \[ \Delta{t}\gamma \int_{\Omega_e} w_i u_i^n d\Omega_e \]
 *
 * Stabilized Buoyancy Force:
 *   \[ \frac{\Delta{t}^2}{2}*\gamma \int_{\Omega_e} (\frac{\partial w_i}{\partial x_k} u_k^n ) u_i^n d\Omega_e \]
 *
 * where \f$ \gamma \f$ is the impermeability constant, \f$ \Delta{t} \f$ is the
 * current time step, \f$ u_i^n \f$ is the i-th component of the previous velocity
 * field and \f$ x_i \f$ is the i-th coordinate.
 *
 ******************************************************************************/
template<typename PhysicsT, typename EvaluationT>
class BrinkmanForces
{
private:
    static constexpr auto mNumDofsPerCell = PhysicsT::mNumDofsPerCell; /*!< number of degrees of freedom per cell */

    static constexpr auto mNumSpatialDims    = PhysicsT::SimplexT::mNumSpatialDims;         /*!< number of spatial dimensions */
    static constexpr auto mNumNodesPerCell   = PhysicsT::SimplexT::mNumNodesPerCell;        /*!< number of nodes per cell */
    static constexpr auto mNumVelDofsPerNode = PhysicsT::SimplexT::mNumMomentumDofsPerNode; /*!< number of momentum dofs per node */

    // set local ad types
    using ResultT  = typename EvaluationT::ResultScalarType; /*!< result FAD type */
    using ConfigT  = typename EvaluationT::ConfigScalarType; /*!< configuration FAD type */
    using ControlT = typename EvaluationT::ControlScalarType; /*!< control FAD type */
    using PrevVelT = typename EvaluationT::PreviousMomentumScalarType; /*!< previous velocity FAD type */

    Plato::Scalar mStabilization = 0.0; /*!< stabilization constant */
    Plato::Scalar mImpermeability = 1.0; /*!< permeability dimensionless number */
    Plato::Scalar mBrinkmanConvexityParam = 0.5;  /*!< brinkman model convexity parameter */

    Plato::DataMap& mDataMap; /*!< output database */
    const Plato::SpatialDomain& mSpatialDomain; /*!< Plato spatial model */
    Plato::LinearTetCubRuleDegreeOne<mNumSpatialDims> mCubatureRule; /*!< cubature rule evaluator */

public:
    /***************************************************************************//**
     * \brief Constructor
     * \param [in] aDomain  spatial domain metadata
     * \param [in] aDataMap output data map
     * \param [in] aInputs  input file metadata
     ******************************************************************************/
    BrinkmanForces
    (const Plato::SpatialDomain & aDomain,
     Plato::DataMap             & aDataMap,
     Teuchos::ParameterList     & aInputs) :
         mDataMap(aDataMap),
         mSpatialDomain(aDomain),
         mCubatureRule(Plato::LinearTetCubRuleDegreeOne<mNumSpatialDims>())
    {
        this->setImpermeability(aInputs);
        mStabilization = Plato::Fluids::stabilization_constant("Momentum Conservation", aInputs);
    }

    /***************************************************************************//**
     * \brief Destructor
     ******************************************************************************/
    ~BrinkmanForces(){}

    /***************************************************************************//**
     * \brief Evaluate Brinkman forces, including stabilized forces if enabled.
     * \param [in] aWorkSets holds input worksets (e.g. states, control, etc)
     * \param [in/out] aResultWS result/output workset
     ******************************************************************************/
    void evaluate
    (const Plato::WorkSets & aWorkSets,
     Plato::ScalarMultiVectorT<ResultT> & aResultWS) const 
    {
        auto tNumCells = mSpatialDomain.numCells();
        if( tNumCells != static_cast<Plato::OrdinalType>(aResultWS.extent(0)) )
        {
            THROWERR(std::string("Number of elements mismatch. Spatial domain and output/result workset ")
                + "have different number of cells. " + "Spatial domain has '" + std::to_string(tNumCells)
                + "' elements and output workset has '" + std::to_string(aResultWS.extent(0)) + "' elements.")
        }

        // set local functors
        Plato::ComputeGradientWorkset<mNumSpatialDims> tComputeGradient;
        Plato::InterpolateFromNodal<mNumSpatialDims, mNumVelDofsPerNode, 0/*offset*/, mNumSpatialDims> tIntrplVectorField;

        // set temporary local worksets
        Plato::ScalarVectorT<ConfigT> tCellVolume("cell weight", tNumCells);
        Plato::ScalarArray3DT<ConfigT> tGradient("cell gradient", tNumCells, mNumNodesPerCell, mNumSpatialDims);

        Plato::ScalarMultiVectorT<ResultT> tBrinkman("cell brinkman forces", tNumCells, mNumSpatialDims);
        Plato::ScalarMultiVectorT<PrevVelT> tPrevVelGP("cell previous velocity", tNumCells, mNumSpatialDims);

        // set input worksets
        auto tControlWS = Plato::metadata<Plato::ScalarMultiVectorT<ControlT>>(aWorkSets.get("control"));
        auto tConfigWS  = Plato::metadata<Plato::ScalarArray3DT<ConfigT>>(aWorkSets.get("configuration"));
        auto tPrevVelWS = Plato::metadata<Plato::ScalarMultiVectorT<PrevVelT>>(aWorkSets.get("previous velocity"));
        auto tCriticalTimeStep = Plato::metadata<Plato::ScalarVector>(aWorkSets.get("critical time step"));

        // transfer member host scalar data to device
        auto tStabilization = mStabilization;
        auto tImpermeability = mImpermeability;
        auto tBrinkmanConvexityParam = mBrinkmanConvexityParam;

        auto tCubWeight = mCubatureRule.getCubWeight();
        auto tBasisFunctions = mCubatureRule.getBasisFunctions();
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
        {
            tComputeGradient(aCellOrdinal, tGradient, tConfigWS, tCellVolume);
            tCellVolume(aCellOrdinal) *= tCubWeight;

            // 1. add brinkman force contribution to residual, R += \Delta{t}\gamma M u^n
            auto tMultiplier = static_cast<Plato::Scalar>(1.0) * tCriticalTimeStep(0);
            tIntrplVectorField(aCellOrdinal, tBasisFunctions, tPrevVelWS, tPrevVelGP);
            ControlT tPenalizedPermeability = Plato::Fluids::brinkman_penalization<mNumNodesPerCell>
                (aCellOrdinal, tImpermeability, tBrinkmanConvexityParam, tControlWS);
            Plato::Fluids::calculate_brinkman_forces<mNumSpatialDims>
                (aCellOrdinal, tPenalizedPermeability, tPrevVelGP, tBrinkman);
            Plato::Fluids::integrate_vector_field<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tBasisFunctions, tCellVolume, tBrinkman, aResultWS, tMultiplier);

            // 2. add stabilizing brinkman force to residual, R += (\frac{\Delta{t}^2}{2}\gamma) M_u u_n
            tMultiplier = tStabilization * static_cast<Plato::Scalar>(0.5) * tCriticalTimeStep(0) * tCriticalTimeStep(0);
            Plato::Fluids::integrate_stabilizing_vector_force<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tCellVolume, tGradient, tPrevVelGP, tBrinkman, aResultWS, tMultiplier);
        }, "brinkman force evaluator");
    }

private:
    /***************************************************************************//**
     * \brief Set impermeability constant.
     * \param [in] aInputs  input file metadata
     ******************************************************************************/
    void setImpermeability
    (Teuchos::ParameterList & aInputs)
    {
	if(Plato::Fluids::is_impermeability_defined(aInputs))
	{
            auto tHyperbolic = aInputs.sublist("Hyperbolic");
            mImpermeability = Plato::teuchos::parse_parameter<Plato::Scalar>("Impermeability Number", "Dimensionless Properties", tHyperbolic);
	}
	else
	{
            auto tHyperbolic = aInputs.sublist("Hyperbolic");
            auto tDaNum = Plato::teuchos::parse_parameter<Plato::Scalar>("Darcy Number", "Dimensionless Properties", tHyperbolic);
            auto tPrNum = Plato::teuchos::parse_parameter<Plato::Scalar>("Prandtl Number", "Dimensionless Properties", tHyperbolic);
            mImpermeability = tPrNum / tDaNum;
	}
    }
};
// class BrinkmanForces






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
 *   \mathcal{R}^n_i(w^h_i) = I^n_i(w^h_i) - F^n_i(w^h_i) - B^n_i(w^h_i) - S_i^n(w^h_i) - E_i^n(w^h_i) = 0.
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
    using PredVelT  = typename EvaluationT::MomentumPredictorScalarType; /*!< predicted velocity FAD type */
    using PrevTempT = typename EvaluationT::PreviousEnergyScalarType; /*!< previous temperature FAD type */

    using AdvectionT  = typename Plato::Fluids::fad_type_t<typename PhysicsT::SimplexT, PrevVelT, ConfigT>; /*!< advection force FAD type */
    using PredStrainT = typename Plato::Fluids::fad_type_t<typename PhysicsT::SimplexT, PredVelT, ConfigT>; /*!< predicted strain rate FAD type */
    using PrevStrainT = typename Plato::Fluids::fad_type_t<typename PhysicsT::SimplexT, PrevVelT, ConfigT>; /*!< previous strain rate FAD type */

    Plato::DataMap& mDataMap; /*!< output database */
    const Plato::SpatialDomain& mSpatialDomain; /*!< Plato spatial model */
    Plato::LinearTetCubRuleDegreeOne<mNumSpatialDims> mCubatureRule; /*!< cubature rule evaluator */

    // set right hand side force evaluators
    std::shared_ptr<Plato::NaturalBCs<mNumSpatialDims, mNumDofsPerNode>> mPrescribedBCs; /*!< prescribed boundary conditions, e.g. tractions */
    std::shared_ptr<Plato::Fluids::BrinkmanForces<PhysicsT,EvaluationT>> mBrinkmanForces; /*!< Brinkman force evaluator */
    std::shared_ptr<Plato::Fluids::ThermalBuoyancy<PhysicsT,EvaluationT>> mThermalBuoyancy; /*!< thermal buoyancy force evaluator */

    // set member scalar data
    Plato::Scalar mTheta = 1.0; /*!< artificial viscous damping */
    Plato::Scalar mViscocity = 1.0; /*!< dimensionless viscocity constant */
    Plato::Scalar mStabilization = 0.0; /*!< stabilization scalar multiplier */

    bool mCalculateBrinkmanForces = false; /*!< indicator to determine if Brinkman forces will be considered in calculations */
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
        this->initialize(aDomain, aDataMap, aInputs);
    }

    /***************************************************************************//**
     * \brief Destructor
     ******************************************************************************/
    virtual ~VelocityPredictorResidual(){}

    /***************************************************************************//**
     * \fn void evaluate
     * \brief Evaluate predictor residual.
     * \param [in] aWorkSets holds input worksets (e.g. states, control, etc)
     * \param [in/out] aResultWS result/output workset
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
        auto tStabilization = mStabilization;

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
            Plato::blas2::scale<mNumDofsPerCell>(aCellOrdinal, tCriticalTimeStep(0), aResultWS);

            // 5. add predicted inertial force to residual, i.e. R += M\bar{u}
            tIntrplVectorField(aCellOrdinal, tBasisFunctions, tPredVelWS, tPredVelGP);
            Plato::Fluids::integrate_vector_field<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tBasisFunctions, tCellVolume, tPredVelGP, aResultWS);

            // 6. add previous inertial force to residual, i.e. R -= M u_n
            Plato::Fluids::integrate_vector_field<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tBasisFunctions, tCellVolume, tPrevVelGP, aResultWS, -1.0);

            // 7. add stabilizing convective term to residual. i.e. R += \frac{\Delta{t}^2}{2}K_{u}u^{n}
            tMultiplier = tStabilization * static_cast<Plato::Scalar>(0.5) * tCriticalTimeStep(0) * tCriticalTimeStep(0);
            Plato::Fluids::integrate_stabilizing_vector_force<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tCellVolume, tGradient, tPrevVelGP, tAdvection, aResultWS, tMultiplier);
        }, "quasi-implicit predicted velocity residual");

        if(mCalculateThermalBuoyancyForces)
        {
            mThermalBuoyancy->evaluate(aWorkSets, aResultWS);
        }

        if(mCalculateBrinkmanForces)
        {
            mBrinkmanForces->evaluate(aWorkSets, aResultWS);
        }
    }

   /***************************************************************************//**
    * \fn void evaluateBoundary
    * \brief Evaluate non-prescribed boundary forces.
    * \param [in]  aSpatialModel holds mesh and entity sets (e.g. node and side sets) metadata
    * \param [in] aWorkSets holds input worksets (e.g. states, control, etc)
    * \param [in/out] aResultWS result/output workset
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
    * \param [in] aWorkSets holds input worksets (e.g. states, control, etc)
    * \param [in/out] aResultWS result/output workset
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
               Plato::blas2::scale<mNumDofsPerCell>(aCellOrdinal, tCriticalTimeStep(0), tTractionWS);
               Plato::blas2::update<mNumDofsPerCell>(aCellOrdinal, -1.0, tTractionWS, 1.0, aResultWS);
           }, "traction force");
       }
   }

private:
   /***************************************************************************//**
    * \fn void initialize
    * \brief Initialize member data.
    * \param [in] aDomain  spatial domain metadata
    * \param [in] aDataMap output data metadata
    * \param [in] aInputs  input file metadata
    ******************************************************************************/
   void initialize
   (const Plato::SpatialDomain & aDomain,
    Plato::DataMap             & aDataMap,
    Teuchos::ParameterList     & aInputs)
   {
       this->setAritificalDamping(aInputs);
       this->setNaturalBoundaryConditions(aInputs);
       mViscocity = Plato::Fluids::calculate_viscosity_constant(aInputs);
       mStabilization = Plato::Fluids::stabilization_constant("Momentum Conservation", aInputs);

       this->setBrinkmanForces(aDomain, aDataMap, aInputs);
       this->setThermalBuoyancyForces(aDomain, aDataMap, aInputs);
   }

   /***************************************************************************//**
    * \fn void setBrinkmanForces
    * \brief Set Brinkman forces if enabled. The Brinkman forces are used to model
    *   a fictitious solid material within a fluid domain. These froces are only used
    *   in density-based topology optimization problems.
    * \param [in] aDomain  spatial domain metadata
    * \param [in] aDataMap output data metadata
    * \param [in] aInputs  input file metadata
    ******************************************************************************/
   void setBrinkmanForces
   (const Plato::SpatialDomain & aDomain,
    Plato::DataMap             & aDataMap,
    Teuchos::ParameterList     & aInputs)
   {
       mCalculateBrinkmanForces = Plato::Fluids::calculate_brinkman_forces(aInputs);
       if(mCalculateBrinkmanForces)
       {
           mBrinkmanForces =
               std::make_shared<Plato::Fluids::BrinkmanForces<PhysicsT,EvaluationT>>(aDomain, aDataMap, aInputs);
       }
   }

   /***************************************************************************//**
    * \fn void setThermalBuoyancyForces
    * \brief Set thermal buoyancy forces if enabled.
    * \param [in] aDomain  spatial domain metadata
    * \param [in] aDataMap output data metadata
    * \param [in] aInputs  input file metadata
    ******************************************************************************/
   void setThermalBuoyancyForces
   (const Plato::SpatialDomain & aDomain,
    Plato::DataMap             & aDataMap,
    Teuchos::ParameterList     & aInputs)
   {
       mCalculateThermalBuoyancyForces = Plato::Fluids::calculate_heat_transfer(aInputs);
       if(mCalculateThermalBuoyancyForces)
       {
           mThermalBuoyancy =
               std::make_shared<Plato::Fluids::ThermalBuoyancy<PhysicsT,EvaluationT>>(aDomain, aDataMap, aInputs);
       }
   }

   /***************************************************************************//**
    * \fn void setAritificalDamping
    * \brief Set artificial viscous damping. This parameter is related to the time
    *   integration scheme.
    * \param [in] aInputs  input file metadata
    ******************************************************************************/
   void setAritificalDamping(Teuchos::ParameterList& aInputs)
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

/***************************************************************************//**
 * \tparam NumNodes   number of nodes on the cell
 * \tparam SpaceDim   spatial dimensions
 * \tparam ConfigT    configuration Forward Automaitc Differentiation (FAD) type
 * \tparam CurPressT  current pressure FAD type
 * \tparam PrevPressT previous pressure FAD type
 * \tparam PressGradT pressure gradient FAD type
 *
 * \fn device_type void calculate_pressure_gradient
 * \brief Calculate pressure gradient, defined as
 *
 * \f[
 *   \frac{\partial p^{n+\theta_2}}{\partial x_i} =
 *     \alpha\left( 1-\theta_2 \right)\frac{partial p^n}{partial x_i}
 *     + \theta_2\frac{\partial\delta{p}}{\partial x_i}
 * \f]
 *
 * where \f$ \delta{p} = p^{n+1} - p^{n} \f$, \f$ x_i \f$ is the i-th coordinate,
 * \f$ \theta_2 \f$ is artificial pressure damping and \f$ \alpha \f$ is a scalar
 * multiplier.
 *
 * \param [in] aCellOrdinal cell/element ordinal
 * \param [in] aTheta       artificial damping
 * \param [in] aGradient    spatial gradient workset
 * \param [in] aCurPress    current pressure workset
 * \param [in] aPrevPress   previous pressure workset
 * \param [in\out] aPressGrad pressure gradient workset
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
// function calculate_pressure_gradient

/***************************************************************************//**
 * \tparam NumNodes   number of nodes on the cell
 * \tparam SpaceDim   spatial dimensions
 * \tparam ConfigT    configuration Forward Automaitc Differentiation (FAD) type
 * \tparam FieldT     scalar field FAD type
 * \tparam FieldGradT scalar field gradient FAD type
 *
 * \fn device_type void calculate_scalar_field_gradient
 * \brief Calculate scalar field gradient, defined as
 *
 * \f[ \frac{\partial p^n}{\partial x_i} = \frac{\partial}{\partial x_i} p^n \f]
 *
 * where \f$ p^{n} \f$ is the pressure field at time step n, \f$ x_i \f$ is the ]
 * i-th coordinate.
 *
 * \param [in] aCellOrdinal cell/element ordinal
 * \param [in] aGradient    spatial gradient workset
 * \param [in] aScalarField scalar field workset
 * \param [in\out] aResult  output/result workset
 *
 ******************************************************************************/
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
// function calculate_scalar_field_gradient


// todo: corrector equation
/***************************************************************************//**
 * \class VelocityCorrectorResidual
 *
 * \tparam PhysicsT    physics type
 * \tparam EvaluationT forward automatic differentiation evaluation type
 *
 * \brief Class responsible for the evaluation of the momentum corrector residual,
 *   defined as
 *
 * \f[
 *   \int_{\Omega_e} w_i u_i^{n+1} d\Omega_e = \int_{\Omega_e} w_i u_i^{\ast} d\Omega_e
 *     - \Delta{t}\int_{\Omega_e} w_i\frac{\partial p^{n+\theta_p}}{\partial x_i}
 *     + \frac{\Delta{t}^2}{2}\int_{\Omega_e}(\frac{\partial w_i}{\partial x_k} u_k^n)
 *       \frac{\partial p^n}{\partial x_i} d\Omega_e.
 * \f]
 *
 * where \f$ u_i \f$ is the i-th component of the velocity field, \f$ u_i^{\ast} \f$
 * is the i-th component of the velocity predictor field, \f$ p^n \f$ is the
 * pressure field, \f$ x_i \f$ is the i-th coordinate and \f$ \theta_p \f$ is
 * the pressure artificial damping parameter.
 *
 ******************************************************************************/
template<typename PhysicsT, typename EvaluationT>
class VelocityCorrectorResidual : public Plato::Fluids::AbstractVectorFunction<PhysicsT, EvaluationT>
{
private:
    static constexpr auto mNumDofsPerNode    = PhysicsT::mNumDofsPerNode; /*!< number of degrees of freedom per node */
    static constexpr auto mNumDofsPerCell    = PhysicsT::mNumDofsPerCell; /*!< number of degrees of freedom per cell */
    static constexpr auto mNumNodesPerCell   = PhysicsT::SimplexT::mNumNodesPerCell; /*!< number of nodes per cell */
    static constexpr auto mNumSpatialDims    = PhysicsT::SimplexT::mNumSpatialDims; /*!< number of spatial dimensions */

    // set local ad types
    using ResultT    = typename EvaluationT::ResultScalarType; /*!< result/output FAD type */
    using ConfigT    = typename EvaluationT::ConfigScalarType; /*!< configuration FAD type */
    using CurVelT    = typename EvaluationT::CurrentMomentumScalarType; /*!< current velocity FAD type */
    using PrevVelT   = typename EvaluationT::PreviousMomentumScalarType; /*!< previous velocity FAD type */
    using PredVelT   = typename EvaluationT::MomentumPredictorScalarType; /*!< predicted velocity FAD type */
    using CurPressT  = typename EvaluationT::CurrentMassScalarType; /*!< current pressure FAD type */
    using PrevPressT = typename EvaluationT::PreviousMassScalarType; /*!< previous pressure FAD type */

    /*!< pressure gradient FAD type */
    using PressGradT = typename Plato::Fluids::fad_type_t<typename PhysicsT::SimplexT, CurPressT, PrevPressT, ConfigT>;

    Plato::DataMap& mDataMap; /*!< output database */
    const Plato::SpatialDomain& mSpatialDomain; /*!< Plato spatial model */
    Plato::LinearTetCubRuleDegreeOne<mNumSpatialDims> mCubatureRule; /*!< integration rule */

    Plato::Scalar mPressureTheta = 1.0; /*!< artificial pressure damping */
    Plato::Scalar mViscosityTheta = 1.0; /*!< artificial viscosity damping */

public:
    /***************************************************************************//**
     * \brief Constructor
     * \param [in] aDomain  spatial domain metadata
     * \param [in] aDataMap output data metadata
     * \param [in] aInputs  input file metadata
     ******************************************************************************/
    VelocityCorrectorResidual
    (const Plato::SpatialDomain & aDomain,
     Plato::DataMap             & aDataMap,
     Teuchos::ParameterList     & aInputs) :
         mDataMap(aDataMap),
         mSpatialDomain(aDomain),
         mCubatureRule(Plato::LinearTetCubRuleDegreeOne<mNumSpatialDims>())
    {
        this->setAritificalPressureDamping(aInputs);
    }

    /***************************************************************************//**
     * \brief Destructor
     ******************************************************************************/
    virtual ~VelocityCorrectorResidual(){}

    /***************************************************************************//**
     * \brief Evaluate Brinkman forces, including stabilized forces if enabled.
     * \param [in] aWorkSets holds input worksets (e.g. states, control, etc)
     * \param [in/out] aResultWS result/output workset
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

        Plato::ScalarMultiVectorT<PressGradT> tPressGradGP("pressure gradient", tNumCells, mNumSpatialDims);
        Plato::ScalarMultiVectorT<CurVelT> tCurVelGP("current velocity at Gauss points", tNumCells, mNumSpatialDims);
        Plato::ScalarMultiVectorT<PrevVelT> tPrevVelGP("previous velocity at Gauss points", tNumCells, mNumSpatialDims);
        Plato::ScalarMultiVectorT<PredVelT> tPredVelGP("predicted velocity at Gauss points", tNumCells, mNumSpatialDims);

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
        auto tPressureTheta = mPressureTheta;
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
            Plato::blas2::scale<mNumDofsPerCell>(aCellOrdinal, tCriticalTimeStep(0), aResultWS);

            // 2. add current delta inertial force to residual, i.e. R += M(u_{n+1} - u_n)
            tIntrplVectorField(aCellOrdinal, tBasisFunctions, tCurVelWS, tCurVelGP);
            Plato::Fluids::integrate_vector_field<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tBasisFunctions, tCellVolume, tCurVelGP, aResultWS);
            tIntrplVectorField(aCellOrdinal, tBasisFunctions, tPrevVelWS, tPrevVelGP);
            Plato::Fluids::integrate_vector_field<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tBasisFunctions, tCellVolume, tPrevVelGP, aResultWS, -1.0);

            // 3. add delta predicted inertial force to residual, i.e. R -= M(\bar{u} - u_n)
            tIntrplVectorField(aCellOrdinal, tBasisFunctions, tPredVelWS, tPredVelGP);
            Plato::Fluids::integrate_vector_field<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tBasisFunctions, tCellVolume, tPredVelGP, aResultWS, -1.0);
            Plato::Fluids::integrate_vector_field<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tBasisFunctions, tCellVolume, tPrevVelGP, aResultWS);
        }, "calculate corrected velocity residual");
    }

    /***************************************************************************//**
     * \brief Evaluate non-prescribed boundary conditions.
     * \param [in] aWorkSets holds input worksets (e.g. states, control, etc)
     * \param [in/out] aResultWS result/output workset
     ******************************************************************************/
    void evaluateBoundary
    (const Plato::SpatialModel & aSpatialModel,
     const Plato::WorkSets & aWorkSets,
     Plato::ScalarMultiVectorT<ResultT> & aResult)
    const override
    { return; /* boundary integral equals zero */ }

    /***************************************************************************//**
     * \brief Evaluate prescribed boundary conditions.
     * \param [in] aWorkSets holds input worksets (e.g. states, control, etc)
     * \param [in/out] aResultWS result/output workset
     ******************************************************************************/
    void evaluatePrescribed
    (const Plato::SpatialModel & aSpatialModel,
     const Plato::WorkSets & aWorkSets,
     Plato::ScalarMultiVectorT<ResultT> & aResult)
    const override
    { return; /* prescribed force integral equals zero */ }

private:
    /***************************************************************************//**
     * \brief Set artificial pressure damping parameter.
     * \param [in] aInputs  input file metadata
     ******************************************************************************/
    void setAritificalPressureDamping(Teuchos::ParameterList& aInputs)
    {
        if(aInputs.isSublist("Time Integration"))
        {
            auto tTimeIntegration = aInputs.sublist("Time Integration");
            mPressureTheta = tTimeIntegration.get<Plato::Scalar>("Pressure Damping", 1.0);
        }
    }
};
// class VelocityCorrectorResidual


/***************************************************************************//**
 * \tparam NumNodes  number of nodes
 * \tparam SpaceDim  spatial dimensions
 * \tparam ConfigT   configuration Forward Automatic Differentiation (FAD) type
 * \tparam PrevVelT  previous velocity FAD type
 * \tparam PrevTempT previous temperatue FAD type
 * \tparam ResultT   result/output FAD type
 *
 * \fn DEVICE_TYPE inline void calculate_convective_forces
 *
 * \brief Calculate convective forces.
 *
 * \param [in] aCellOrdinal cell/element ordinal
 * \param [in] aGradient    spatial gradient workset
 * \param [in] aPrevVelGP   previous velocity at the Gauss points
 * \param [in] aPrevTemp    previous temperature workset
 * \param [in] aMultiplier  scalar multiplier
 * \param [in\out] aResult  output/result workset
 *
 ******************************************************************************/
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
// function calculate_convective_forces


/***************************************************************************//**
 * \tparam NumNodes  number of nodes
 * \tparam SpaceDim  spatial dimensions
 * \tparam ConfigT   configuration Forward Automatic Differentiation (FAD) type
 * \tparam PrevVelT  previous velocity FAD type
 * \tparam PrevTempT previous temperature FAD type
 * \tparam ResultT   result/output FAD type
 *
 * \fn DEVICE_TYPE inline void integrate_scalar_field
 *
 * \brief Integrate scalar field, defined as
 *
 *   \f[ \int_{\Omega_e} w^h F d\Omega_e \f]
 *
 * where \f$ w^h \f$ are the test functions and \f$ F \f$ is the scalar field.
 *
 * \param [in] aCellOrdinal    cell/element ordinal
 * \param [in] aBasisFunctions basis functions
 * \param [in] aCellVolume     cell volume workset
 * \param [in] aField          scalar field workset
 * \param [in] aMultiplier     scalar multiplier
 * \param [in\out] aResult     output/result workset
 *
 ******************************************************************************/
template
<Plato::OrdinalType NumNodes,
 typename ConfigT,
 typename SourceT,
 typename ResultT,
 typename ScalarT>
DEVICE_TYPE inline void
integrate_scalar_field
(const Plato::OrdinalType & aCellOrdinal,
 const Plato::ScalarVector & aBasisFunctions,
 const Plato::ScalarVectorT<ConfigT> & aCellVolume,
 const Plato::ScalarVectorT<SourceT> & aField,
 const Plato::ScalarMultiVectorT<ResultT> & aResult,
       ScalarT aMultiplier)
{
    for(Plato::OrdinalType tNode = 0; tNode < NumNodes; tNode++)
    {
        aResult(aCellOrdinal, tNode) += aMultiplier * aBasisFunctions(tNode) *
            aField(aCellOrdinal) * aCellVolume(aCellOrdinal);
    }
}
// function integrate_scalar_field

/***************************************************************************//**
 * \tparam NumNodes  number of nodes
 * \tparam SpaceDim  spatial dimensions
 * \tparam ConfigT   configuration Forward Automatic Differentiation (FAD) type
 * \tparam FluxT     flux FAD type
 * \tparam ScalarT   scalar multiplier FAD type
 * \tparam ResultT   result/output FAD type
 *
 * \fn DEVICE_TYPE inline void calculate_flux_divergence
 *
 * \brief Calculate flux divergence, defined as
 *
 *   \f[ \int_{\Omega_e} \frac{\partial}{\partial x_i} F_i d\Omega_e \f]
 *
 * where \f$ w^h \f$ are the test functions and \f$ F_i \f$ is the i-th flux.
 *
 * \param [in] aCellOrdinal ell/element ordinal
 * \param [in] aGradient    spatial gradient
 * \param [in] aCellVolume  cell volume workset
 * \param [in] aFlux        flux
 * \param [in] aMultiplier  scalar multiplier
 * \param [in\out] aResult  output/result workset
 *
 ******************************************************************************/
template
<Plato::OrdinalType NumNodes,
 Plato::OrdinalType SpaceDim,
 typename ConfigT,
 typename FluxT,
 typename ResultT,
 typename ScalarT>
DEVICE_TYPE inline void
calculate_flux_divergence
(const Plato::OrdinalType & aCellOrdinal,
 const Plato::ScalarArray3DT<ConfigT> & aGradient,
 const Plato::ScalarVectorT<ConfigT> & aCellVolume,
 const Plato::ScalarMultiVectorT<FluxT> & aFlux,
 const Plato::ScalarMultiVectorT<ResultT> & aResult,
 const ScalarT & aMultiplier)
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
// function calculate_flux_divergence


/***************************************************************************//**
 * \tparam NumNodes number of nodes
 * \tparam SpaceDim spatial dimensions
 * \tparam FluxT    flux FAD type
 * \tparam ConfigT  configuration Forward Automatic Differentiation (FAD) type
 * \tparam StateT   state FAD type
 *
 * \fn DEVICE_TYPE inline void calculate_flux
 *
 * \brief Calculate flux divergence, defined as
 *
 *   \f[ \int_{\Omega_e} \frac{\partial}{\partial x_i} F d\Omega_e \f]
 *
 * where \f$ w^h \f$ are the test functions and \f$ F \f$ is a scalar field.
 *
 * \param [in] aCellOrdinal ell/element ordinal
 * \param [in] aGradient    spatial gradient
 * \param [in] aScalarField scalar field
 * \param [in\out] aFlux output/result workset
 *
 ******************************************************************************/
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
 const Plato::ScalarMultiVectorT<StateT> & aScalarField,
 const Plato::ScalarMultiVectorT<FluxT> & aFlux)
{
    for(Plato::OrdinalType tNode = 0; tNode < NumNodes; tNode++)
    {
        for(Plato::OrdinalType tDim = 0; tDim < SpaceDim; tDim++)
        {
            aFlux(aCellOrdinal, tDim) += aGradient(aCellOrdinal, tNode, tDim) * aScalarField(aCellOrdinal, tNode);
        }
    }
}
// function calculate_flux


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
template
<Plato::OrdinalType NumNodesPerCell,
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
// function penalize_thermal_diffusivity



/***************************************************************************//**
 * \tparam NumNodesPerCell number of nodes per cell/element (integer)
 * \tparam ControlT control Forward Automatic Differentiation (FAD) evaluation type
 *
 * \fn DEVICE_TYPE inline ControlT penalize_heat_source_constant
 *
 * \brief Penalize heat source constant. This function is only needed for
 *   density-based topology optimization problems.
 *
 * \param [in] aCellOrdinal cell/element ordinal
 * \param [in] aConstant    heat source constant
 * \param [in] aPenalty     penalty exponent used for density-based penalty model
 * \param [in] aControl     control workset
 *
 * \return penalized heat source constant
 ******************************************************************************/
template
<Plato::OrdinalType NumNodesPerCell,
 typename ControlT>
DEVICE_TYPE inline ControlT
penalize_heat_source_constant
(const Plato::OrdinalType & aCellOrdinal,
 const Plato::Scalar & aConstant,
 const Plato::Scalar & aPenalty,
 const Plato::ScalarMultiVectorT<ControlT> & aControl)
{
    ControlT tDensity = Plato::cell_density<NumNodesPerCell>(aCellOrdinal, aControl);
    ControlT tPenalizedDensity = pow(tDensity, aPenalty);
    auto tPenalizedProperty = (static_cast<Plato::Scalar>(1) - tPenalizedDensity) * aConstant;
    return tPenalizedProperty;
}
// function penalize_heat_source_constant

/***************************************************************************//**
 * \tparam NumNodes number of nodes per cell/element (integer)
 * \tparam SpaceDim spatial dimensions (integer)
 * \tparam ResultT  result Forward Automatic Differentiation (FAD) evaluation type
 * \tparam ConfigT  configuration FAD evaluation type
 * \tparam PrevVelT previous velocity FAD evaluation type
 * \tparam StabT    stabilization FAD evaluation type
 *
 * \fn DEVICE_TYPE inline void integrate_stabilizing_scalar_forces
 *
 * \brief Integrate stabilized scalar field.
 *
 * \param [in] aCellOrdinal cell/element ordinal
 * \param [in] aCellVolume  cell/element volume workset
 * \param [in] aGradient    spatial gradient
 * \param [in] aPrevVelGP   previous velocity at Gauss points
 * \param [in] aStabForce   stabilizing force workset
 * \param [in] aMultiplier  scalar multiplier
 * \param [in/out] aResult  result/output workset
 *
 ******************************************************************************/
template
<Plato::OrdinalType NumNodes,
 Plato::OrdinalType SpaceDim,
 typename ResultT,
 typename ConfigT,
 typename PrevVelT,
 typename StabT>
DEVICE_TYPE inline void
integrate_stabilizing_scalar_forces
(const Plato::OrdinalType & aCellOrdinal,
 const Plato::ScalarVectorT<ConfigT> & aCellVolume,
 const Plato::ScalarArray3DT<ConfigT> & aGradient,
 const Plato::ScalarMultiVectorT<PrevVelT> & aPrevVelGP,
 const Plato::ScalarVectorT<StabT> & aStabForce,
 const Plato::ScalarMultiVectorT<ResultT> & aResult,
 Plato::Scalar aMultiplier = 1.0)
 {
    for(Plato::OrdinalType tNode = 0; tNode < NumNodes; tNode++)
    {
        for(Plato::OrdinalType tDimK = 0; tDimK < SpaceDim; tDimK++)
        {
            aResult(aCellOrdinal, tNode) += aMultiplier * ( ( aGradient(aCellOrdinal, tNode, tDimK) *
                aPrevVelGP(aCellOrdinal, tDimK) ) * aStabForce(aCellOrdinal) ) * aCellVolume(aCellOrdinal);
        }
    }
 }
// function integrate_stabilizing_scalar_forces

/***************************************************************************//**
 * \class VelocityCorrectorResidual
 *
 * \tparam PhysicsT    physics type
 * \tparam EvaluationT Forward Automatic Differentiation (FAD) evaluation type
 *
 * \brief Abstract class used to defined cell/element volume integrals.
 *
 ******************************************************************************/
template<typename PhysicsT, typename EvaluationT>
class AbstractCellVolumeIntegral
{
private:
    using ResultT = typename EvaluationT::ResultScalarType; /*!< result FAD evaluation type */

public:
    virtual ~AbstractCellVolumeIntegral(){}
    virtual void evaluate(const Plato::WorkSets & aWorkSets, Plato::ScalarMultiVectorT<ResultT> & aResultWS) const = 0;
};
// class AbstractCellVolumeIntegral


namespace SIMP
{

/***************************************************************************//**
 * \class CellThermalVolumeIntegral
 *
 * \tparam PhysicsT    physics type
 * \tparam EvaluationT Forward Automatic Differentiation (FAD) evaluation type
 *
 * \brief Derived class responsible for the evaluation of the internal thermal
 *   forces. This implementation is only used for density-based topology
 *   optimization problems.
 *
 ******************************************************************************/
template<typename PhysicsT, typename EvaluationT>
class CellThermalVolumeIntegral : public Plato::Fluids::AbstractCellVolumeIntegral<PhysicsT,EvaluationT>
{
private:
    static constexpr auto mNumDofsPerNode = PhysicsT::mNumDofsPerNode; /*!< number of degrees of freedom per node */
    static constexpr auto mNumDofsPerCell = PhysicsT::mNumDofsPerCell; /*!< number of degrees of freedom per cell */

    static constexpr auto mNumSpatialDims       = PhysicsT::SimplexT::mNumSpatialDims; /*!< number of spatial dimensions */
    static constexpr auto mNumNodesPerCell      = PhysicsT::SimplexT::mNumNodesPerCell; /*!< number of nodes per cell/element */
    static constexpr auto mNumVelDofsPerCell    = PhysicsT::SimplexT::mNumMomentumDofsPerCell; /*!< number of momentum degrees of freedom per cell */
    static constexpr auto mNumTempDofsPerCell   = PhysicsT::SimplexT::mNumEnergyDofsPerCell; /*!< number of energy degrees of freedom per node */
    static constexpr auto mNumVelDofsPerNode    = PhysicsT::SimplexT::mNumMomentumDofsPerNode; /*!< number of momentum degrees of freedom per node */
    static constexpr auto mNumTempDofsPerNode   = PhysicsT::SimplexT::mNumEnergyDofsPerNode; /*!< number of energy degrees of freedom per node */
    static constexpr auto mNumConfigDofsPerCell = PhysicsT::SimplexT::mNumConfigDofsPerCell; /*!< number of configuration degrees of freedom per cell */

    // set local ad type
    using ResultT   = typename EvaluationT::ResultScalarType; /*!< result FAD evaluation type */
    using ConfigT   = typename EvaluationT::ConfigScalarType; /*!< configuration FAD evaluation type */
    using ControlT  = typename EvaluationT::ControlScalarType; /*!< control FAD evaluation type */
    using CurVelT   = typename EvaluationT::CurrentMomentumScalarType; /*!< current momentum FAD evaluation type */
    using CurTempT  = typename EvaluationT::CurrentEnergyScalarType; /*!< current energy FAD evaluation type */
    using PrevTempT = typename EvaluationT::PreviousEnergyScalarType; /*!< previous energy FAD evaluation type */

    using CurFluxT = typename Plato::Fluids::fad_type_t<typename PhysicsT::SimplexT, CurTempT, ConfigT>; /*!< current flux FAD evaluation type */
    using PrevFluxT = typename Plato::Fluids::fad_type_t<typename PhysicsT::SimplexT, PrevTempT, ConfigT>; /*!< previous flux FAD evaluation type */
    using ConvectionT = typename Plato::Fluids::fad_type_t<typename PhysicsT::SimplexT, PrevTempT, CurVelT, ConfigT>; /*!< convection FAD evaluation type */

    Plato::Scalar mStabilization = 0.0; /*!< stabilization scalar multiplier */
    Plato::Scalar mArtificialDamping = 1.0; /*!< artificial temperature damping - damping is a byproduct from time integration scheme */
    Plato::Scalar mHeatSourceConstant = 0.0; /*!< heat source constant */
    Plato::Scalar mThermalConductivity = 1.0; /*!< thermal conductivity */
    Plato::Scalar mCharacteristicLength = 0.0; /*!< characteristic lenght */
    Plato::Scalar mReferenceTemperature = 1.0; /*!< reference temperature */
    Plato::Scalar mEffectiveConductivity = 1.0; /*!< effective conductivity */
    Plato::Scalar mThermalDiffusivityRatio = 1.0; /*!< thermal diffusivity ratio, e.g. solid diffusivity / fluid diffusivity */
    Plato::Scalar mHeatSourcePenaltyExponent = 3.0; /*!< exponent used for heat source penalty model */
    Plato::Scalar mThermalDiffusivityPenaltyExponent = 3.0; /*!< exponent used for internal flux penalty model */

    Plato::DataMap& mDataMap; /*!< output database */
    const Plato::SpatialDomain& mSpatialDomain; /*!< spatial domain metadata */
    Plato::LinearTetCubRuleDegreeOne<mNumSpatialDims> mCubatureRule; /*!< cubature integration rule */

public:
    /***************************************************************************//**
     * \brief Constructor
     * \param [in] aDomain  spatial domain metadata
     * \param [in] aDataMap output database metadata
     * \param [in] aInputs  input file metadata
     ******************************************************************************/
    CellThermalVolumeIntegral
    (const Plato::SpatialDomain & aDomain,
     Plato::DataMap             & aDataMap,
     Teuchos::ParameterList     & aInputs) :
        mDataMap(aDataMap),
        mSpatialDomain(aDomain),
        mCubatureRule(Plato::LinearTetCubRuleDegreeOne<mNumSpatialDims>())
    {
        this->setSourceTerm(aInputs);
        this->setPenaltyModelParameters(aInputs);
        this->setThermalDiffusivityRatio(aInputs);
        this->setAritificalDiffusiveDamping(aInputs);
	    mEffectiveConductivity = Plato::Fluids::calculate_effective_conductivity(aInputs);
        mStabilization = Plato::Fluids::stabilization_constant("Energy Conservation", aInputs);
    }

    /***************************************************************************//**
     * \brief Destructor
     ******************************************************************************/
    ~CellThermalVolumeIntegral(){}

    /***************************************************************************//**
     * \brief Evaluate internal thermal forces. This implementation is only used for
     *   density-based topology optimization problems.
     * \param [in] aWorkSets holds input worksets (e.g. states, control, etc)
     * \param [in/out] aResultWS result/output workset
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
        Plato::ScalarVectorT<ResultT> tHeatSource("prescribed heat source", tNumCells);
        Plato::blas1::fill(mHeatSourceConstant, tHeatSource);

        // set local data
        Plato::ScalarVectorT<ConfigT> tCellVolume("cell weight", tNumCells);
        Plato::ScalarArray3DT<ConfigT> tGradient("cell gradient", tNumCells, mNumNodesPerCell, mNumSpatialDims);

        Plato::ScalarVectorT<CurTempT> tCurTempGP("current temperature at Gauss points", tNumCells);
        Plato::ScalarVectorT<PrevTempT> tPrevTempGP("previous temperature at Gauss points", tNumCells);
        Plato::ScalarVectorT<ConvectionT> tConvection("convection", tNumCells);

        Plato::ScalarMultiVectorT<CurFluxT> tCurThermalFlux("current thermal flux", tNumCells, mNumSpatialDims);
        Plato::ScalarMultiVectorT<PrevFluxT> tPrevThermalFlux("previous thermal flux", tNumCells, mNumSpatialDims);
        Plato::ScalarMultiVectorT<CurVelT> tCurVelGP("current velocity at Gauss points", tNumCells, mNumVelDofsPerNode);

        // set local functors
        Plato::ComputeGradientWorkset<mNumSpatialDims> tComputeGradient;
        Plato::InterpolateFromNodal<mNumSpatialDims, mNumTempDofsPerNode> tIntrplScalarField;
        Plato::InterpolateFromNodal<mNumSpatialDims, mNumVelDofsPerNode, 0, mNumSpatialDims> tIntrplVectorField;

        // set input state worksets
        auto tConfigWS   = Plato::metadata<Plato::ScalarArray3DT<ConfigT>>(aWorkSets.get("configuration"));
        auto tCurVelWS   = Plato::metadata<Plato::ScalarMultiVectorT<CurVelT>>(aWorkSets.get("current velocity"));
        auto tControlWS  = Plato::metadata<Plato::ScalarMultiVectorT<ControlT>>(aWorkSets.get("control"));
        auto tCurTempWS  = Plato::metadata<Plato::ScalarMultiVectorT<CurTempT>>(aWorkSets.get("current temperature"));
        auto tPrevTempWS = Plato::metadata<Plato::ScalarMultiVectorT<PrevTempT>>(aWorkSets.get("previous temperature"));
        auto tCriticalTimeStep = Plato::metadata<Plato::ScalarVector>(aWorkSets.get("critical time step"));

        // transfer member data to device
        auto tRefTemp = mReferenceTemperature;
        auto tCharLength = mCharacteristicLength;
        auto tThermalCond = mThermalConductivity;
        auto tStabilization = mStabilization;
        auto tArtificialDamping = mArtificialDamping;
        auto tEffConductivity = mEffectiveConductivity;
        auto tThermalDiffusivityRatio = mThermalDiffusivityRatio;
        auto tHeatSourcePenaltyExponent = mHeatSourcePenaltyExponent;
        auto tThermalDiffusivityPenaltyExponent = mThermalDiffusivityPenaltyExponent;

        auto tCubWeight = mCubatureRule.getCubWeight();
        auto tBasisFunctions = mCubatureRule.getBasisFunctions();
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
        {
            tComputeGradient(aCellOrdinal, tGradient, tConfigWS, tCellVolume);
            tCellVolume(aCellOrdinal) = tCellVolume(aCellOrdinal) * tCubWeight;

            // 1. Penalize diffusivity ratio with element density
            ControlT tPenalizedDiffusivityRatio = Plato::Fluids::penalize_thermal_diffusivity<mNumNodesPerCell>
                (aCellOrdinal, tThermalDiffusivityRatio, tThermalDiffusivityPenaltyExponent, tControlWS);
            ControlT tPenalizedEffConductivity = tEffConductivity * tPenalizedDiffusivityRatio;

            // 2. add current diffusive force contribution to residual, i.e. R += \theta_3 K T^{n+1}, 
            Plato::Fluids::calculate_flux<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tGradient, tCurTempWS, tCurThermalFlux);
            ControlT tMultiplierControlT = tArtificialDamping * tPenalizedEffConductivity;
            Plato::Fluids::calculate_flux_divergence<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tGradient, tCellVolume, tCurThermalFlux, aResultWS, tMultiplierControlT);

            // 3. add previous diffusive force contribution to residual, i.e. R -= (\theta_3-1) K T^n
            Plato::Fluids::calculate_flux<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tGradient, tPrevTempWS, tPrevThermalFlux);
            tMultiplierControlT = (tArtificialDamping - static_cast<Plato::Scalar>(1.0)) * tPenalizedEffConductivity;
            Plato::Fluids::calculate_flux_divergence<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tGradient, tCellVolume, tPrevThermalFlux, aResultWS, -tMultiplierControlT);

            // 4. add previous heat source contribution to residual, i.e. R -= \alpha Q^n
            auto tHeatSrcDimlessConstant = ( tCharLength * tCharLength ) / (tThermalCond * tRefTemp);
            ControlT tPenalizedDimlessHeatSrcConstant = Plato::Fluids::penalize_heat_source_constant<mNumNodesPerCell>
                (aCellOrdinal, tHeatSrcDimlessConstant, tHeatSourcePenaltyExponent, tControlWS);
            Plato::Fluids::integrate_scalar_field<mNumTempDofsPerCell>
                (aCellOrdinal, tBasisFunctions, tCellVolume, tHeatSource, aResultWS, -tPenalizedDimlessHeatSrcConstant);

            // 5. add current convective force contribution to residual, i.e. R += C(u^{n+1}) T^n
            tIntrplVectorField(aCellOrdinal, tBasisFunctions, tCurVelWS, tCurVelGP);
            Plato::Fluids::calculate_convective_forces<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tGradient, tCurVelGP, tPrevTempWS, tConvection);
            Plato::Fluids::integrate_scalar_field<mNumTempDofsPerCell>
                (aCellOrdinal, tBasisFunctions, tCellVolume, tConvection, aResultWS, 1.0);
            Plato::blas2::scale<mNumDofsPerCell>(aCellOrdinal, tCriticalTimeStep(0), aResultWS);

            // 6. add stabilizing force contribution to residual, i.e. R += C_u(u^{n+1}) T^n - Q_u(u^{n+1})
            auto tScalar = tStabilization * static_cast<Plato::Scalar>(0.5) * tCriticalTimeStep(0) * tCriticalTimeStep(0);
            Plato::Fluids::integrate_stabilizing_scalar_forces<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tCellVolume, tGradient, tCurVelGP, tConvection, aResultWS, tScalar);
            tScalar = tStabilization * tHeatSrcDimlessConstant *
                static_cast<Plato::Scalar>(0.5) * tCriticalTimeStep(0) * tCriticalTimeStep(0);
            Plato::Fluids::integrate_stabilizing_scalar_forces<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tCellVolume, tGradient, tCurVelGP, tHeatSource, aResultWS, -tScalar);
        }, "energy conservation residual");
    }

private:
    /***************************************************************************//**
     * \brief Set heat source parameters.
     * \param [in] aInputs input file metadata.
     ******************************************************************************/
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

            this->setThermalConductivity(aInputs);
            this->setCharacteristicLength(aInputs);
        }
    }

    /***************************************************************************//**
     * \brief Set thermal conductivity.
     * \param [in] aInputs input file metadata.
     ******************************************************************************/
    void setThermalConductivity(Teuchos::ParameterList &aInputs)
    {
        auto tMaterialName = mSpatialDomain.getMaterialName();
        Plato::teuchos::is_material_defined(tMaterialName, aInputs);
        auto tMaterial = aInputs.sublist("Material Models").sublist(tMaterialName);
        mThermalConductivity = Plato::teuchos::parse_parameter<Plato::Scalar>("Thermal Conductivity", "Thermal Properties", tMaterial);
        Plato::is_positive_finite_number(mThermalConductivity, "Thermal Conductivity");
    }

    /***************************************************************************//**
     * \brief Set thermal diffusivity ratio.
     * \param [in] aInputs input file metadata.
     ******************************************************************************/
    void setThermalDiffusivityRatio(Teuchos::ParameterList &aInputs)
    {
        auto tMaterialName = mSpatialDomain.getMaterialName();
        Plato::teuchos::is_material_defined(tMaterialName, aInputs);
        auto tMaterial = aInputs.sublist("Material Models").sublist(tMaterialName);
        mThermalDiffusivityRatio = Plato::teuchos::parse_parameter<Plato::Scalar>("Thermal Diffusivity Ratio", "Thermal Properties", tMaterial);
        Plato::is_positive_finite_number(mThermalDiffusivityRatio, "Thermal Diffusivity Ratio");
    }

    /***************************************************************************//**
     * \brief Set characteristic length.
     * \param [in] aInputs input file metadata.
     ******************************************************************************/
    void setCharacteristicLength
    (Teuchos::ParameterList & aInputs)
    {
        if(aInputs.isSublist("Hyperbolic") == false)
        {
            THROWERR("'Hyperbolic' Parameter List is not defined.")
        }
        auto tHyperbolic = aInputs.sublist("Hyperbolic");
        mCharacteristicLength = Plato::teuchos::parse_parameter<Plato::Scalar>("Characteristic Length", "Dimensionless Properties", tHyperbolic);
    }

    /***************************************************************************//**
     * \brief Set artificial diffusive damping.
     * \param [in] aInputs input file metadata.
     ******************************************************************************/
    void setAritificalDiffusiveDamping
    (Teuchos::ParameterList& aInputs)
    {
        if(aInputs.isSublist("Time Integration"))
        {
            auto tTimeIntegration = aInputs.sublist("Time Integration");
            mArtificialDamping = tTimeIntegration.get<Plato::Scalar>("Diffusive Damping", 1.0);
        }
    }

    /***************************************************************************//**
     * \brief Set penalty parameters for density penalization model.
     * \param [in] aInputs input file metadata.
     ******************************************************************************/
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
                mThermalDiffusivityPenaltyExponent = tPenaltyFuncList.get<Plato::Scalar>("Thermal Diffusion Penalty Exponent", 3.0);
            }
        }
    }
};
// class CellThermalVolumeIntegral

}
// namespace SIMP


/***************************************************************************//**
 * \class CellThermalVolumeIntegral
 *
 * \tparam PhysicsT    physics type
 * \tparam EvaluationT Forward Automatic Differentiation (FAD) evaluation type
 *
 * \brief Derived class responsible for the evaluation of the internal thermal
 *   forces. This implementation is used for forward simulations. In addition,
 *   this implementation can be used for level-set based topology optimization
 *   problems and parametric CAD shape optimization problems.
 *
 ******************************************************************************/
template<typename PhysicsT, typename EvaluationT>
class CellThermalVolumeIntegral : public Plato::Fluids::AbstractCellVolumeIntegral<PhysicsT,EvaluationT>
{
private:
    static constexpr auto mNumDofsPerNode = PhysicsT::mNumDofsPerNode; /*!< number of degrees of freedom per node */
    static constexpr auto mNumDofsPerCell = PhysicsT::mNumDofsPerCell; /*!< number of degrees of freedom per cell */

    static constexpr auto mNumSpatialDims       = PhysicsT::SimplexT::mNumSpatialDims; /*!< number of spatial dimensions */
    static constexpr auto mNumNodesPerCell      = PhysicsT::SimplexT::mNumNodesPerCell; /*!< number of nodes per cell */
    static constexpr auto mNumVelDofsPerCell    = PhysicsT::SimplexT::mNumMomentumDofsPerCell; /*!< number of momentum degrees of freedom per cell */
    static constexpr auto mNumTempDofsPerCell   = PhysicsT::SimplexT::mNumEnergyDofsPerCell; /*!< number of energy degrees of freedom per node */
    static constexpr auto mNumVelDofsPerNode    = PhysicsT::SimplexT::mNumMomentumDofsPerNode; /*!< number of momentum degrees of freedom per node */
    static constexpr auto mNumTempDofsPerNode   = PhysicsT::SimplexT::mNumEnergyDofsPerNode; /*!< number of energy degrees of freedom per node */
    static constexpr auto mNumConfigDofsPerCell = PhysicsT::SimplexT::mNumConfigDofsPerCell; /*!< number of configuration degrees of freedom per cell */

    // set local ad type
    using ResultT   = typename EvaluationT::ResultScalarType; /*!< result FAD evaluation type */
    using ConfigT   = typename EvaluationT::ConfigScalarType; /*!< configuration FAD evaluation type */
    using ControlT  = typename EvaluationT::ControlScalarType; /*!< control FAD evaluation type */
    using CurVelT   = typename EvaluationT::CurrentMomentumScalarType; /*!< current velocity FAD evaluation type */
    using CurTempT  = typename EvaluationT::CurrentEnergyScalarType; /*!< current temperature FAD evaluation type */
    using PrevTempT = typename EvaluationT::PreviousEnergyScalarType; /*!< previous temperature FAD evaluation type */

    using CurFluxT = typename Plato::Fluids::fad_type_t<typename PhysicsT::SimplexT, CurTempT, ConfigT>; /*!< current flux FAD evaluation type */
    using PrevFluxT = typename Plato::Fluids::fad_type_t<typename PhysicsT::SimplexT, PrevTempT, ConfigT>; /*!< previous flux FAD evaluation type */
    using ConvectionT = typename Plato::Fluids::fad_type_t<typename PhysicsT::SimplexT, PrevTempT, CurVelT, ConfigT>; /*!< convection FAD evaluation type */

    Plato::Scalar mStabilization = 0.0; /*!< stabilization scalar multiplier */
    Plato::Scalar mArtificialDamping = 1.0; /*!< artificial temperature damping - damping is a byproduct from time integration scheme */
    Plato::Scalar mHeatSourceConstant = 0.0; /*!< heat source constant */
    Plato::Scalar mThermalConductivity = 1.0; /*!< thermal conductivity */
    Plato::Scalar mCharacteristicLength = 0.0; /*!< characteristic length */
    Plato::Scalar mReferenceTemperature = 1.0; /*!< reference temperature */
    Plato::Scalar mEffectiveConductivity = 1.0; /*!< effective conductivity */

    Plato::DataMap& mDataMap; /*!< output database */
    const Plato::SpatialDomain& mSpatialDomain; /*!< spatial domain metadata */
    Plato::LinearTetCubRuleDegreeOne<mNumSpatialDims> mCubatureRule; /*!< cubature integration rule */

public:
    /***************************************************************************//**
     * \brief Constructor
     * \param [in] aDomain  spatial domain metadata
     * \param [in] aDataMap output database metadata
     * \param [in] aInputs  input file metadata
     ******************************************************************************/
    CellThermalVolumeIntegral
    (const Plato::SpatialDomain & aDomain,
     Plato::DataMap             & aDataMap,
     Teuchos::ParameterList     & aInputs) :
        mDataMap(aDataMap),
        mSpatialDomain(aDomain),
        mCubatureRule(Plato::LinearTetCubRuleDegreeOne<mNumSpatialDims>())
    {
        this->setSourceTerm(aInputs);
        this->setAritificalDiffusiveDamping(aInputs);
	    mEffectiveConductivity = Plato::Fluids::calculate_effective_conductivity(aInputs);
        mStabilization = Plato::Fluids::stabilization_constant("Energy Conservation", aInputs);
    }

    /***************************************************************************//**
     * \brief Destructor
     ******************************************************************************/
    ~CellThermalVolumeIntegral(){}

    /***************************************************************************//**
     * \brief Evaluate internal thermal forces.
     * \param [in] aWorkSets holds input worksets (e.g. states, control, etc)
     * \param [in/out] aResultWS result/output workset
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
        Plato::ScalarVectorT<ResultT> tHeatSource("prescribed heat source", tNumCells);
        Plato::blas1::fill(mHeatSourceConstant, tHeatSource);

        // set local data
        Plato::ScalarVectorT<ConfigT> tCellVolume("cell weight", tNumCells);
        Plato::ScalarArray3DT<ConfigT> tGradient("cell gradient", tNumCells, mNumNodesPerCell, mNumSpatialDims);

        Plato::ScalarVectorT<CurTempT> tCurTempGP("current temperature at Gauss points", tNumCells);
        Plato::ScalarVectorT<PrevTempT> tPrevTempGP("previous temperature at Gauss points", tNumCells);
        Plato::ScalarVectorT<ConvectionT> tConvection("convection", tNumCells);

        Plato::ScalarMultiVectorT<CurFluxT> tCurThermalFlux("current thermal flux", tNumCells, mNumSpatialDims);
        Plato::ScalarMultiVectorT<PrevFluxT> tPrevThermalFlux("previous thermal flux", tNumCells, mNumSpatialDims);
        Plato::ScalarMultiVectorT<CurVelT> tCurVelGP("current velocity at Gauss points", tNumCells, mNumVelDofsPerNode);

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
        auto tTheta           = mArtificialDamping;
        auto tRefTemp         = mReferenceTemperature;
        auto tCharLength      = mCharacteristicLength;
        auto tThermalCond     = mThermalConductivity;
        auto tStabilization   = mStabilization;
        auto tEffConductivity = mEffectiveConductivity;

        auto tCubWeight = mCubatureRule.getCubWeight();
        auto tBasisFunctions = mCubatureRule.getBasisFunctions();
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
        {
            tComputeGradient(aCellOrdinal, tGradient, tConfigWS, tCellVolume);
            tCellVolume(aCellOrdinal) = tCellVolume(aCellOrdinal) * tCubWeight;

            // 1. add previous diffusive force contribution to residual, i.e. R -= (\theta_3-1) K T^n
            auto tMultiplier = (tTheta - static_cast<Plato::Scalar>(1));
            Plato::Fluids::calculate_flux<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tGradient, tPrevTempWS, tPrevThermalFlux);
            Plato::blas2::scale<mNumSpatialDims>(aCellOrdinal, tEffConductivity, tPrevThermalFlux);
            Plato::Fluids::calculate_flux_divergence<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tGradient, tCellVolume, tPrevThermalFlux, aResultWS, -tMultiplier);

            // 2. add current convective force contribution to residual, i.e. R += C(u^{n+1}) T^n
            tIntrplVectorField(aCellOrdinal, tBasisFunctions, tCurVelWS, tCurVelGP);
            Plato::Fluids::calculate_convective_forces<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tGradient, tCurVelGP, tPrevTempWS, tConvection);
            Plato::Fluids::integrate_scalar_field<mNumTempDofsPerCell>
                (aCellOrdinal, tBasisFunctions, tCellVolume, tConvection, aResultWS, 1.0);

            // 3. add current diffusive force contribution to residual, i.e. R += \theta_3 K T^{n+1}
            Plato::Fluids::calculate_flux<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tGradient, tCurTempWS, tCurThermalFlux);
            Plato::blas2::scale<mNumSpatialDims>(aCellOrdinal, tEffConductivity, tCurThermalFlux);
            Plato::Fluids::calculate_flux_divergence<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tGradient, tCellVolume, tCurThermalFlux, aResultWS, tTheta);

            // 4. add previous heat source contribution to residual, i.e. R -= \alpha Q^n
            auto tHeatSourceConstant = ( tCharLength * tCharLength ) / (tThermalCond * tRefTemp);
            Plato::Fluids::integrate_scalar_field<mNumTempDofsPerCell>
                (aCellOrdinal, tBasisFunctions, tCellVolume, tHeatSource, aResultWS, -tHeatSourceConstant);
            Plato::blas2::scale<mNumDofsPerCell>(aCellOrdinal, tCriticalTimeStep(0), aResultWS);

            // 5. add stabilizing force contribution to residual, i.e. R += C_u(u^{n+1}) T^n - Q_u(u^{n+1})
            tMultiplier = tStabilization * static_cast<Plato::Scalar>(0.5) * tCriticalTimeStep(0) * tCriticalTimeStep(0);
            Plato::Fluids::integrate_stabilizing_scalar_forces<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tCellVolume, tGradient, tCurVelGP, tConvection, aResultWS, tMultiplier);
            tMultiplier = tStabilization * tHeatSourceConstant * static_cast<Plato::Scalar>(0.5) * tCriticalTimeStep(0) * tCriticalTimeStep(0);
            Plato::Fluids::integrate_stabilizing_scalar_forces<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tCellVolume, tGradient, tCurVelGP, tHeatSource, aResultWS, -tMultiplier);
        }, "energy conservation residual");
    }

private:
    /***************************************************************************//**
     * \brief Set heat source parameters.
     * \param [in] aInputs input file metadata.
     ******************************************************************************/
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

    /***************************************************************************//**
     * \brief Set thermal properties.
     * \param [in] aInputs input file metadata.
     ******************************************************************************/
    void setThermalProperties
    (Teuchos::ParameterList & aInputs)
    {
        if(aInputs.isSublist("Heat Source"))
        {
            auto tMaterialName = mSpatialDomain.getMaterialName();
            Plato::teuchos::is_material_defined(tMaterialName, aInputs);
            auto tMaterial = aInputs.sublist("Material Models").sublist(tMaterialName);
            auto tThermalPropBlock = std::string("Thermal Properties");
            mThermalConductivity = Plato::teuchos::parse_parameter<Plato::Scalar>("Thermal Conductivity", tThermalPropBlock, tMaterial);
	        Plato::is_positive_finite_number(mThermalConductivity, "Thermal Conductivity");
        }
    }

    /***************************************************************************//**
     * \brief Set characteristic length.
     * \param [in] aInputs input file metadata.
     ******************************************************************************/
    void setCharacteristicLength
    (Teuchos::ParameterList & aInputs)
    {
        if(aInputs.isSublist("Hyperbolic") == false)
        {
            THROWERR("'Hyperbolic' Parameter List is not defined.")
        }
        auto tHyperbolic = aInputs.sublist("Hyperbolic");
        mCharacteristicLength = Plato::teuchos::parse_parameter<Plato::Scalar>("Characteristic Length", "Dimensionless Properties", tHyperbolic);
    }

    /***************************************************************************//**
     * \brief Set artificial damping.
     * \param [in] aInputs input file metadata.
     ******************************************************************************/
    void setAritificalDiffusiveDamping
    (Teuchos::ParameterList& aInputs)
    {
        if(aInputs.isSublist("Time Integration"))
        {
            auto tTimeIntegration = aInputs.sublist("Time Integration");
            mArtificialDamping = tTimeIntegration.get<Plato::Scalar>("Diffusive Damping", 1.0);
        }
    }
};
// class CellThermalVolumeIntegral


/***************************************************************************//**
 * \strut CellVolumeIntegralFactory
 *
 * \brief Factory for internal force integrals for computational fluid dynamics
 *   applications.
 *
 ******************************************************************************/
struct CellVolumeIntegralFactory
{

/***************************************************************************//**
 * \tparam PhysicsT    Physics type
 * \tparam EvaluationT Forward Automatic Differentiation (FAD) evaluation type
 *
 * \fn inline std::shared_ptr<AbstractCellVolumeIntegral> createThermalVolumeIntegral
 *
 * \brief Return shared pointer to an abstract cell volume integral instance.
 *
 * \param [in] aDomain  spatial domain metadata
 * \param [in] aDataMap output database
 * \param [in] aInputs  input file metadata
 *
 ******************************************************************************/
template <typename PhysicsT, typename EvaluationT>
inline std::shared_ptr<Plato::Fluids::AbstractCellVolumeIntegral<PhysicsT, EvaluationT>>
createThermalVolumeIntegral
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
        return ( std::make_shared<Plato::Fluids::SIMP::CellThermalVolumeIntegral<PhysicsT, EvaluationT>>(aDomain, aDataMap, aInputs) );
    }
    else if( tLowerScenario == "analysis" || tLowerScenario == "levelset to" )
    {
        return ( std::make_shared<Plato::Fluids::CellThermalVolumeIntegral<PhysicsT, EvaluationT>>(aDomain, aDataMap, aInputs) );
    }
    else
    {
        THROWERR(std::string("Scenario '") + tScenario + "' is not supported. Options are 1) Analysis, 2) Density TO or 3) Levelset TO.")
    }
}

};
// struct CellVolumeIntegralFactory


/***************************************************************************//**
 * \class TemperatureResidual
 *
 * \tparam PhysicsT    Physics type
 * \tparam EvaluationT Forward Automatic Differentiation (FAD) evaluation type
 *
 * \brief Class responsible for the evaluation of the energy residual.
 *
 ******************************************************************************/
template<typename PhysicsT, typename EvaluationT>
class TemperatureResidual : public Plato::Fluids::AbstractVectorFunction<PhysicsT, EvaluationT>
{
private:
    static constexpr auto mNumDofsPerNode = PhysicsT::mNumDofsPerNode; /*!< number of degrees of freedom per node */
    static constexpr auto mNumDofsPerCell = PhysicsT::mNumDofsPerCell; /*!< number of degrees of freedom per cell */
    static constexpr auto mNumSpatialDims = PhysicsT::SimplexT::mNumSpatialDims; /*!< number of spatial dimensions */

    // set local ad type
    using ResultT   = typename EvaluationT::ResultScalarType; /*!< result FAD evaluation type */
    using ConfigT   = typename EvaluationT::ConfigScalarType; /*!< configuration FAD evaluation type */
    using ControlT  = typename EvaluationT::ControlScalarType; /*!< control FAD evaluation type */
    using PrevTempT = typename EvaluationT::PreviousEnergyScalarType;  /*!< previous energy FAD evaluation type */

    const Plato::SpatialDomain& mSpatialDomain; /*!< spatial domain metadata */
    std::shared_ptr<Plato::NaturalBCs<mNumSpatialDims, mNumDofsPerNode>> mHeatFlux; /*!< natural boundary condition evaluator */
    std::shared_ptr<Plato::Fluids::AbstractCellVolumeIntegral<PhysicsT,EvaluationT>> mVolumeIntegral; /*!< volume integral evaluator */

public:
    /***************************************************************************//**
     * \brief Constructor
     * \param [in] aDomain  spatial domain metadata
     * \param [in] aDataMap output database metadata
     * \param [in] aInputs  input file metadata
     ******************************************************************************/
    TemperatureResidual
    (const Plato::SpatialDomain & aDomain,
     Plato::DataMap             & aDataMap,
     Teuchos::ParameterList     & aInputs) :
         mSpatialDomain(aDomain)
    {
        this->setNaturalBoundaryConditions(aInputs);

        Plato::Fluids::CellVolumeIntegralFactory tFactory;
        mVolumeIntegral = tFactory.createThermalVolumeIntegral<PhysicsT,EvaluationT>(aDomain,aDataMap,aInputs);
    }

    /***************************************************************************//**
     * \brief Destructor
     ******************************************************************************/
    virtual ~TemperatureResidual(){}

    /***************************************************************************//**
     * \fn void evaluate
     * \brief Evaluate predictor residual.
     * \param [in] aWorkSets holds input worksets (e.g. states, control, etc)
     * \param [in/out] aResultWS result/output workset
     ******************************************************************************/
    void evaluate
    (const Plato::WorkSets & aWorkSets,
     Plato::ScalarMultiVectorT<ResultT> & aResultWS)
    const override
    {
        mVolumeIntegral->evaluate(aWorkSets, aResultWS);
    }

    /***************************************************************************//**
     * \fn void evaluateBoundary
     * \brief Evaluate predictor residual.
     * \param [in] aWorkSets holds input worksets (e.g. states, control, etc)
     * \param [in/out] aResultWS result/output workset
     ******************************************************************************/
    void evaluateBoundary
    (const Plato::SpatialModel & aSpatialModel,
     const Plato::WorkSets & aWorkSets,
     Plato::ScalarMultiVectorT<ResultT> & aResult)
    const override
    { return;  }

    /***************************************************************************//**
     * \fn void evaluatePrescribed
     * \brief Evaluate prescribed boundary forces.
     * \param [in]  aSpatialModel holds mesh and entity sets (e.g. node and side sets) metadata
     * \param [in] aWorkSets holds input worksets (e.g. states, control, etc)
     * \param [in/out] aResultWS result/output workset
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
            Plato::ScalarMultiVectorT<ResultT> tHeatFluxWS("heat flux", tNumCells, mNumDofsPerCell);
            mHeatFlux->get( aSpatialModel, tPrevTempWS, tControlWS, tConfigWS, tHeatFluxWS );

            auto tCriticalTimeStep = Plato::metadata<Plato::ScalarVector>(aWorkSets.get("critical time step"));
            Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
            {
                Plato::blas2::scale<mNumDofsPerCell>(aCellOrdinal, tCriticalTimeStep(0), tHeatFluxWS);
                Plato::blas2::update<mNumDofsPerCell>(aCellOrdinal, -1.0, tHeatFluxWS, 1.0, aResultWS);
            }, "heat flux contribution");
        }
    }

private:
    /***************************************************************************//**
     * \fn void setNaturalBoundaryConditions
     * \brief Set natural boundary conditions. The boundary conditions are set based
     *   on the information available in the input file.
     * \param [in] aInputs  input file metadata
     ******************************************************************************/
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


/***************************************************************************//**
 * \fn device_type void integrate_divergence_operator
 *
 * \tparam NumNodes number of nodes on cell/element (integer)
 * \tparam SpaceDim spatial dimensions (integer)
 * \tparam ConfigT  configuration FAD evaluation type
 * \tparam PrevVelT previous velocity FAD evaluation type
 * \tparam aResult  result/output FAD evaluation type
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
 * \param [in] aCellOrdinal    cell/element ordinal
 * \param [in] aBasisFunctions basis functions
 * \param [in] aGradient       spatial gradient workset
 * \param [in] aCellVolume     cell/element volume workset
 * \param [in] aPrevVel        previous velocity workset
 * \param [in] aMultiplier     scalar multiplier
 * \param [in/out] aResult     result/output workset
 *
 ******************************************************************************/
template<Plato::OrdinalType NumNodes,
         Plato::OrdinalType SpaceDim,
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
    for(Plato::OrdinalType tNode = 0; tNode < NumNodes; tNode++)
    {
        for(Plato::OrdinalType tDim = 0; tDim < SpaceDim; tDim++)
        {
            aResult(aCellOrdinal, tNode) += aMultiplier * aCellVolume(aCellOrdinal) *
                aBasisFunctions(tNode) * aGradient(aCellOrdinal, tNode, tDim) * aPrevVel(aCellOrdinal, tDim);
        }
    }
}
// function integrate_divergence_operator

/***************************************************************************//**
 * \fn device_type void integrate_laplacian_operator
 *
 * \tparam NumNodes number of nodes on cell/element (integer)
 * \tparam SpaceDim spatial dimensions (integer)
 * \tparam ConfigT  configuration FAD evaluation type
 * \tparam FieldT   field FAD evaluation type
 * \tparam aResult  result/output FAD evaluation type
 *
 * \brief Integrate Laplacian operator, defined as
 *
 * \f[
 *   \alpha\int_{\Omega} \frac{\partial v^h}{\partial x_i}\frac{\partial p^n}{\partial x_i} d\Omega
 * \f]
 *
 * where \f$ v^h \f$ is the test function, \f$ p^{n} \f$ is a scalar field at
 * time step n, \f$ x_i \f$ is the i-th coordinate and \f$ \alpha \f$ denotes
 * a scalar multiplier.
 *
 * \param [in] aCellOrdinal    cell/element ordinal
 * \param [in] aGradient       spatial gradient workset
 * \param [in] aCellVolume     cell/element volume workset
 * \param [in] aField          vector field workset
 * \param [in] aMultiplier     scalar multiplier
 * \param [in/out] aResult     result/output workset
 *
 ******************************************************************************/
template<Plato::OrdinalType NumNodes,
         Plato::OrdinalType SpaceDim,
         typename ConfigT,
         typename FieldT,
         typename ResultT>
DEVICE_TYPE inline void
integrate_laplacian_operator
(const Plato::OrdinalType & aCellOrdinal,
 const Plato::ScalarArray3DT<ConfigT> & aGradient,
 const Plato::ScalarVectorT<ConfigT> & aCellVolume,
 const Plato::ScalarMultiVectorT<FieldT> & aField,
 const Plato::ScalarMultiVectorT<ResultT> & aResult,
 Plato::Scalar aMultiplier = 1.0)
{
    for(Plato::OrdinalType tNode = 0; tNode < NumNodes; tNode++)
    {
        for(Plato::OrdinalType tDim = 0; tDim < SpaceDim; tDim++)
        {
            aResult(aCellOrdinal, tNode) += aMultiplier * aCellVolume(aCellOrdinal) *
                aGradient(aCellOrdinal, tNode, tDim) * aField(aCellOrdinal, tDim);
        }
    }
}
// function integrate_laplacian_operator





/***************************************************************************//**
 * \class MomentumSurfaceForces
 *
 * \tparam PhysicsT    Physics type
 * \tparam EvaluationT Forward Automatic Differentiation (FAD) evaluation type
 *
 * \brief Class responsible for the evaluation of the surface momentum forces.
 *   This surface integral is evaluated during the calculation of the pressure
 *   residual (i.e. mass conservation equation).
 *
 * \f[
 *   \alpha\int_{\Gamma_e} v^h \left( u_i^n n_i \right) d\Gamma_e
 * \f]
 *
 * where \f$ v^h \f$ is the test function, \f$ u_i^n \f$ is the i-th velocity
 * component at time step n, \f$\f$ is the i-th unit normal component and
 * \f$ \alpha \f$ is a scalar multiplier.
 *
 ******************************************************************************/
template<typename PhysicsT, typename EvaluationT>
class MomentumSurfaceForces
{
private:
    static constexpr auto mNumSpatialDims       = PhysicsT::SimplexT::mNumSpatialDims; /*!< number of spatial dimensions */
    static constexpr auto mNumNodesPerFace      = PhysicsT::SimplexT::mNumNodesPerFace; /*!< number of nodes per face */
    static constexpr auto mNumVelDofsPerNode    = PhysicsT::SimplexT::mNumMomentumDofsPerNode; /*!< number of momentum dofs per node */
    static constexpr auto mNumSpatialDimsOnFace = PhysicsT::SimplexT::mNumSpatialDimsOnFace; /*!< number of spatial dimensions on face */

    // forward automatic differentiation types
    using ResultT = typename EvaluationT::ResultScalarType; /*!< result FAD evaluation type */
    using ConfigT = typename EvaluationT::ConfigScalarType; /*!< configuration FAD evaluation type */
    using PrevVelT = typename EvaluationT::PreviousMomentumScalarType; /*!< previous momentum FAD evaluation type */

    const std::string mEntitySetName; /*!< entity set name, defined by the surfaces where Dirichlet boundary conditions are applied */
    const Plato::SpatialDomain& mSpatialDomain; /*!< spatial domain metadata */
    Plato::LinearTetCubRuleDegreeOne<mNumSpatialDimsOnFace> mSurfaceCubatureRule; /*!< surface cubature integration rule */

public:
    /***************************************************************************//**
     * \brief Constructor
     * \param [in] aDomain        spatial domain metadata
     * \param [in] aEntitySetName entity set name (e.g. side set name)
     ******************************************************************************/
    MomentumSurfaceForces
    (const Plato::SpatialDomain & aDomain,
     const std::string & aEntitySetName) :
         mEntitySetName(aEntitySetName),
         mSpatialDomain(aDomain)
    {
    }

    /***************************************************************************//**
     * \fn void operator()
     * \brief Evaluate surface integral.
     * \param [in] aWorkSets   holds input worksets (e.g. states, control, etc)
     * \param [in] aMultiplier scalar multiplier (default = 1.0)
     * \param [in/out] aResultWS result/output workset
     ******************************************************************************/
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
        Plato::ScalarMultiVectorT<PrevVelT> tPrevVelGP("previous velocity", tNumCells, mNumVelDofsPerNode);

        // set input state worksets
        auto tConfigWS = Plato::metadata<Plato::ScalarArray3DT<ConfigT>>(aWorkSets.get("configuration"));
        auto tPrevVelWS = Plato::metadata<Plato::ScalarMultiVectorT<PrevVelT>>(aWorkSets.get("previous velocity"));
	    auto tCriticalTimeStep = Plato::metadata<Plato::ScalarVector>(aWorkSets.get("critical time step"));

        // evaluate integral
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

              // project velocity field onto surface
	          Plato::project_vector_field_onto_surface<mNumSpatialDims,mNumNodesPerFace>
                 (tCellOrdinal, tSurfaceBasisFunctions, tLocalNodeOrd, tPrevVelWS, tPrevVelGP);

	          auto tMultiplier = aMultiplier / tCriticalTimeStep(0);
              for( Plato::OrdinalType tNode = 0; tNode < mNumNodesPerFace; tNode++ )
              {
                  auto tLocalCellNode = tLocalNodeOrd[tNode];
                  for( Plato::OrdinalType tDim = 0; tDim < mNumSpatialDims; tDim++ )
                  {
                      aResult(tCellOrdinal, tLocalCellNode) += tMultiplier * tUnitNormalVec(tDim) *
                          tPrevVelGP(tCellOrdinal, tDim) * tSurfaceBasisFunctions(tNode) * tSurfaceAreaTimesCubWeight;
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
 * \brief Evaluate pressure equation residual, defined by
 *
 * \f[
 *   \mathcal{R}^n(v^h) =
 *       \Delta{t}\alpha_p\alpha_u\int_{\Omega_e}\frac{\partial v^h}{\partial x_i}\frac{\partial \Delta{p}}{\partial x_i} d\Omega_e
 *     - \int_{\Omega_e}\frac{\partial v^h}{x_i}u_i^n d\Omega_e
 *     - \alpha_p\int_{\Omega_e}\frac{\partial v^h}{x_i}u_i^{\ast} d\Omega_e
 *     + \Delta{t}\alpha_p\int_{\Omega_e}\frac{\partial v^h}{\partial x_i}\frac{\partial p^n}{\partial x_i} d\Omega_e
 *     - \int_{\Gamma_e} v^h \left( u_i^n n_i \right) d\Gamma_e = 0
 * \f]
 *
 * The surface integral defined above is the simplified form from:
 *
 * \f[
 *   \int_{\Gamma_e} v^h n_i \left( u_i^n + \alpha_p\left( u_i^{n+1} - u_i^{n} \right) \right) d\Gamma_e
 * \f]
 *
 * since \f$ \alpha_p \f$ is alway set to one.
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

    // set local FAD types
    using ResultT    = typename EvaluationT::ResultScalarType; /*!< result FAD evaluation type */
    using ConfigT    = typename EvaluationT::ConfigScalarType; /*!< configuration FAD evaluation type */
    using PrevVelT   = typename EvaluationT::PreviousMomentumScalarType; /*!< previous velocity FAD evaluation type */
    using PredVelT   = typename EvaluationT::MomentumPredictorScalarType; /*!< predicted velocity FAD evaluation type */
    using CurPressT  = typename EvaluationT::CurrentMassScalarType; /*!< current pressure FAD evaluation type */
    using PrevPressT = typename EvaluationT::PreviousMassScalarType; /*!< previous pressure FAD evaluation type */

    using CurPressGradT = typename Plato::Fluids::fad_type_t<typename PhysicsT::SimplexT, CurPressT, ConfigT>; /*!< current pressure gradient FAD evaluation type */
    using PrevPressGradT = typename Plato::Fluids::fad_type_t<typename PhysicsT::SimplexT, PrevPressT, ConfigT>; /*!< previous pressure gradient FAD evaluation type */

    Plato::DataMap& mDataMap; /*!< output database */
    const Plato::SpatialDomain& mSpatialDomain; /*!< Plato spatial model */
    Plato::LinearTetCubRuleDegreeOne<mNumSpatialDims> mCubatureRule; /*!< integration rule */

    // artificial damping
    Plato::Scalar mPressDamping = 1.0; /*!< artificial pressure damping */
    Plato::Scalar mMomentumDamping = 1.0; /*!< artificial momentum/velocity damping */
    Plato::Scalar mSurfaceMomentumDamping = 0.31; /*!< artificial surface momentum/velocity damping */

    // surface integral
    using MomentumForces = Plato::Fluids::MomentumSurfaceForces<PhysicsT, EvaluationT>; /*!< local surface momentum force type */
    std::unordered_map<std::string, std::shared_ptr<MomentumForces>> mMomentumBCs; /*!< list of surface momentum forces */

public:
    /***************************************************************************//**
     * \brief Constructor
     * \param [in] aDomain  spatial domain metadata
     * \param [in] aDataMap output database metadata
     * \param [in] aInputs  input file metadata
     ******************************************************************************/
    PressureResidual
    (const Plato::SpatialDomain & aDomain,
     Plato::DataMap             & aDataMap,
     Teuchos::ParameterList     & aInputs) :
         mDataMap(aDataMap),
         mSpatialDomain(aDomain),
         mCubatureRule(Plato::LinearTetCubRuleDegreeOne<mNumSpatialDims>())
    {
        this->setAritificalDamping(aInputs);
        this->setSurfaceBoundaryIntegrals(aInputs);
    }

    /***************************************************************************//**
     * \brief Constructor
     * \param [in] aDomain  spatial domain metadata
     * \param [in] aDataMap output database metadata
     ******************************************************************************/
    PressureResidual
    (const Plato::SpatialDomain & aDomain,
     Plato::DataMap             & aDataMap) :
         mDataMap(aDataMap),
         mSpatialDomain(aDomain),
         mCubatureRule(Plato::LinearTetCubRuleDegreeOne<mNumSpatialDims>())
    {
    }

    /***************************************************************************//**
     * \brief Destructor
     ******************************************************************************/
    virtual ~PressureResidual(){}

    /***************************************************************************//**
     * \brief Evaluate internal forces.
     * \param [in] aWorkSets holds input worksets (e.g. states, control, etc)
     * \param [in/out] aResultWS result/output workset
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
            Plato::blas2::scale<mNumPressDofsPerCell>(aCellOrdinal, tMultiplier, tRightHandSide);

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
            Plato::blas2::update<mNumPressDofsPerCell>(aCellOrdinal, -1.0, tRightHandSide, 1.0, aResultWS);
        }, "calculate continuity residual");
    }

    /***************************************************************************//**
     * \brief Evaluate non-prescribed boundary conditions.
     * \param [in] aWorkSets holds input worksets (e.g. states, control, etc)
     * \param [in/out] aResultWS result/output workset
     ******************************************************************************/
    void evaluateBoundary
    (const Plato::SpatialModel & aSpatialModel,
     const Plato::WorkSets & aWorkSets,
     Plato::ScalarMultiVectorT<ResultT> & aResultWS) const override
    {
        for(auto& tPair : mMomentumBCs)
        {
            tPair.second->operator()(aWorkSets, aResultWS, mSurfaceMomentumDamping);
        }
    }

    /***************************************************************************//**
     * \fn void evaluateBoundary
     * \brief Evaluate predictor residual.
     * \param [in] aWorkSets holds input worksets (e.g. states, control, etc)
     * \param [in/out] aResultWS result/output workset
     ******************************************************************************/
    void evaluatePrescribed
    (const Plato::SpatialModel & aSpatialModel,
     const Plato::WorkSets & aWorkSets,
     Plato::ScalarMultiVectorT<ResultT> & aResult) const override
    { return; }

private:
    /***************************************************************************//**
     * \brief Set artifical pressure and momentum damping.
     * \param [in] aInputs input file metadata.
     ******************************************************************************/
    void setAritificalDamping(Teuchos::ParameterList &aInputs)
    {
        if(aInputs.isSublist("Time Integration"))
        {
            auto tTimeIntegration = aInputs.sublist("Time Integration");
            mPressDamping = tTimeIntegration.get<Plato::Scalar>("Pressure Damping", 1.0);
            mMomentumDamping = tTimeIntegration.get<Plato::Scalar>("Momentum Damping", 1.0);
        }

        if(aInputs.isSublist("Hyperbolic") == false)
        {
            THROWERR("'Hyperbolic' Parameter List is not defined.")
        }
        auto tHyperbolic = aInputs.sublist("Hyperbolic");
        if(tHyperbolic.isSublist("Mass Conservation"))
        {
            auto tMassConservation = tHyperbolic.sublist("Mass Conservation");
            mSurfaceMomentumDamping = tMassConservation.get<Plato::Scalar>("Surface Momentum Damping", 0.32);
        }
    }

    /***************************************************************************//**
     * \brief Set surface boundary integrals.
     * \param [in] aInputs input file metadata.
     ******************************************************************************/
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

    // forward automatic differentiation (FAD) evaluation types
    using ResidualEvalT      = typename Plato::Fluids::Evaluation<typename PhysicsT::SimplexT>::Residual; /*!< residual FAD evaluation type */
    using GradConfigEvalT    = typename Plato::Fluids::Evaluation<typename PhysicsT::SimplexT>::GradConfig; /*!< gradient wrt configuration FAD evaluation type */
    using GradControlEvalT   = typename Plato::Fluids::Evaluation<typename PhysicsT::SimplexT>::GradControl; /*!< gradient wrt control FAD evaluation type */
    using GradCurVelEvalT    = typename Plato::Fluids::Evaluation<typename PhysicsT::SimplexT>::GradCurMomentum; /*!< gradient wrt current momentum FAD evaluation type */
    using GradPrevVelEvalT   = typename Plato::Fluids::Evaluation<typename PhysicsT::SimplexT>::GradPrevMomentum; /*!< gradient wrt previous momentum FAD evaluation type */
    using GradCurTempEvalT   = typename Plato::Fluids::Evaluation<typename PhysicsT::SimplexT>::GradCurEnergy; /*!< gradient wrt current energy FAD evaluation type */
    using GradPrevTempEvalT  = typename Plato::Fluids::Evaluation<typename PhysicsT::SimplexT>::GradPrevEnergy; /*!< gradient wrt previous energy FAD evaluation type */
    using GradCurPressEvalT  = typename Plato::Fluids::Evaluation<typename PhysicsT::SimplexT>::GradCurMass; /*!< gradient wrt current mass FAD evaluation type */
    using GradPrevPressEvalT = typename Plato::Fluids::Evaluation<typename PhysicsT::SimplexT>::GradPrevMass; /*!< gradient wrt previous mass FAD evaluation type */
    using GradPredictorEvalT = typename Plato::Fluids::Evaluation<typename PhysicsT::SimplexT>::GradPredictor; /*!< gradient wrt momentum predictor FAD evaluation type */

    // element residual vector function types
    using ResidualFuncT      = std::shared_ptr<Plato::Fluids::AbstractVectorFunction<PhysicsT, ResidualEvalT>>; /*!< vector function of type residual */
    using GradConfigFuncT    = std::shared_ptr<Plato::Fluids::AbstractVectorFunction<PhysicsT, GradConfigEvalT>>; /*!< vector function of type gradient wrt configuration */
    using GradControlFuncT   = std::shared_ptr<Plato::Fluids::AbstractVectorFunction<PhysicsT, GradControlEvalT>>; /*!< vector function of type gradient wrt control */
    using GradCurVelFuncT    = std::shared_ptr<Plato::Fluids::AbstractVectorFunction<PhysicsT, GradCurVelEvalT>>; /*!< vector function of type gradient wrt current velocity */
    using GradPrevVelFuncT   = std::shared_ptr<Plato::Fluids::AbstractVectorFunction<PhysicsT, GradPrevVelEvalT>>; /*!< vector function of type gradient wrt previous velocity */
    using GradCurTempFuncT   = std::shared_ptr<Plato::Fluids::AbstractVectorFunction<PhysicsT, GradCurTempEvalT>>; /*!< vector function of type gradient wrt current temperature */
    using GradPrevTempFuncT  = std::shared_ptr<Plato::Fluids::AbstractVectorFunction<PhysicsT, GradPrevTempEvalT>>; /*!< vector function of type gradient wrt previous temperature */
    using GradCurPressFuncT  = std::shared_ptr<Plato::Fluids::AbstractVectorFunction<PhysicsT, GradCurPressEvalT>>; /*!< vector function of type gradient wrt current pressure */
    using GradPrevPressFuncT = std::shared_ptr<Plato::Fluids::AbstractVectorFunction<PhysicsT, GradPrevPressEvalT>>; /*!< vector function of type gradient wrt previous pressure */
    using GradPredictorFuncT = std::shared_ptr<Plato::Fluids::AbstractVectorFunction<PhysicsT, GradPredictorEvalT>>; /*!< vector function of type gradient wrt velocity predictor */

    // element vector functions per element block, i.e. domain
    std::unordered_map<std::string, ResidualFuncT>      mResidualFuncs; /*!< vector function list of type residual */
    std::unordered_map<std::string, GradConfigFuncT>    mGradConfigFuncs; /*!< vector function list of type gradient wrt configuration */
    std::unordered_map<std::string, GradControlFuncT>   mGradControlFuncs; /*!< vector function list of type gradient wrt control */
    std::unordered_map<std::string, GradCurVelFuncT>    mGradCurVelFuncs; /*!< vector function list of type gradient wrt current velocity */
    std::unordered_map<std::string, GradPrevVelFuncT>   mGradPrevVelFuncs; /*!< vector function list of type gradient wrt previous velocity */
    std::unordered_map<std::string, GradCurTempFuncT>   mGradCurTempFuncs; /*!< vector function list of type gradient wrt current temperature */
    std::unordered_map<std::string, GradPrevTempFuncT>  mGradPrevTempFuncs; /*!< vector function list of type gradient wrt previous temperature */
    std::unordered_map<std::string, GradCurPressFuncT>  mGradCurPressFuncs; /*!< vector function list of type gradient wrt current pressure */
    std::unordered_map<std::string, GradPrevPressFuncT> mGradPrevPressFuncs; /*!< vector function list of type gradient wrt previous pressure */
    std::unordered_map<std::string, GradPredictorFuncT> mGradPredictorFuncs; /*!< vector function list of type gradient wrt velocity predictor */

    Plato::DataMap& mDataMap; /*!< output database */
    const Plato::SpatialModel& mSpatialModel; /*!< spatial model metadata - owns mesh metadata for all the domains, i.e. element blocks */
    Plato::LocalOrdinalMaps<PhysicsT> mLocalOrdinalMaps; /*!< local-to-global ordinal maps */
    Plato::VectorEntryOrdinal<mNumSpatialDims,mNumDofsPerNode> mStateOrdinalsMap; /*!< local-to-global ordinal vector field map */

public:
    /**************************************************************************//**
    * \brief Constructor
    * \param [in] aTag     vector function tag/type
    * \param [in] aModel   struct that contains the mesh, meshsets, domains, etc.
    * \param [in] aDataMap output database
    * \param [in] aInputs  input file metadata
    ******************************************************************************/
    VectorFunction
    (const std::string            & aTag,
     const Plato::SpatialModel    & aModel,
           Plato::DataMap         & aDataMap,
           Teuchos::ParameterList & aInputs) :
        mSpatialModel(aModel),
        mDataMap(aDataMap),
        mLocalOrdinalMaps(aModel.Mesh),
        mStateOrdinalsMap(&aModel.Mesh)
    {
        this->initialize(aTag, aDataMap, aInputs);
    }

    /**************************************************************************//**
    * \fn integer getNumSpatialDims
    * \brief Return number of spatial dimensions.
    * \return number of spatial dimensions (integer)
    ******************************************************************************/
    decltype(mNumSpatialDims) getNumSpatialDims() const
    {
        return mNumSpatialDims;
    }

    /**************************************************************************//**
    * \fn integer getNumDofsPerCell
    * \brief Return number of degrees of freedom per cell.
    * \return degrees of freedom per cell (integer)
    ******************************************************************************/
    decltype(mNumDofsPerCell) getNumDofsPerCell() const
    {
        return mNumDofsPerCell;
    }

    /**************************************************************************//**
    * \fn integer getNumDofsPerNode
    * \brief Return number of degrees of freedom per node.
    * \return degrees of freedom per node (integer)
    ******************************************************************************/
    decltype(mNumDofsPerNode) getNumDofsPerNode() const
    {
        return mNumDofsPerNode;
    }

    /**************************************************************************//**
    * \fn Plato::ScalarVector value
    * \brief Return vector function residual.
    * \param [in] aControls control variables
    * \param [in] aPrimal   primal state database
    * \return vector function residual
    ******************************************************************************/
    Plato::ScalarVector value
    (const Plato::ScalarVector & aControls,
     const Plato::Primal       & aPrimal)
    const
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
                (tDomain, aControls, aPrimal, mLocalOrdinalMaps, tInputWorkSets);

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
                (tNumCells, aControls, aPrimal, mLocalOrdinalMaps, tInputWorkSets);

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

    /**************************************************************************//**
    * \fn Teuchos::RCP<Plato::CrsMatrixType> gradientConfig
    * \brief Return gradient of residual with respet to (wrt) configuration variables (i.e. Jacobian).
    * \param [in] aControls control variables
    * \param [in] aPrimal   primal state database
    * \return Jacobian wrt configuration
    ******************************************************************************/
    Teuchos::RCP<Plato::CrsMatrixType>
    gradientConfig
    (const Plato::ScalarVector & aControls,
     const Plato::Primal       & aPrimal)
    const
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
                (tDomain, aControls, aPrimal, mLocalOrdinalMaps, tInputWorkSets);

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
                (tNumCells, aControls, aPrimal, mLocalOrdinalMaps, tInputWorkSets);

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

    /**************************************************************************//**
    * \fn Teuchos::RCP<Plato::CrsMatrixType> gradientControl
    * \brief Return gradient of residual with respet to (wrt) control variables (i.e. Jacobian).
    * \param [in] aControls control variables
    * \param [in] aPrimal   primal state database
    * \return Jacobian wrt control
    ******************************************************************************/
    Teuchos::RCP<Plato::CrsMatrixType>
    gradientControl
    (const Plato::ScalarVector & aControls,
     const Plato::Primal       & aVariables)
    const
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

    /**************************************************************************//**
    * \fn Teuchos::RCP<Plato::CrsMatrixType> gradientPredictor
    * \brief Return gradient of residual with respet to (wrt) predictor (i.e. Jacobian).
    * \param [in] aControls control variables
    * \param [in] aPrimal   primal state database
    * \return Jacobian wrt predictor
    ******************************************************************************/
    Teuchos::RCP<Plato::CrsMatrixType>
    gradientPredictor
    (const Plato::ScalarVector & aControls,
     const Plato::Primal       & aPrimal)
    const
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
                (tDomain, aControls, aPrimal, mLocalOrdinalMaps, tInputWorkSets);

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
                (tNumCells, aControls, aPrimal, mLocalOrdinalMaps, tInputWorkSets);

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

    /**************************************************************************//**
    * \fn Teuchos::RCP<Plato::CrsMatrixType> gradientPreviousVel
    * \brief Return gradient of residual with respet to (wrt) previous velocity (i.e. Jacobian).
    * \param [in] aControls control variables
    * \param [in] aPrimal   primal state database
    * \return Jacobian wrt previous velocity
    ******************************************************************************/
    Teuchos::RCP<Plato::CrsMatrixType>
    gradientPreviousVel
    (const Plato::ScalarVector & aControls,
     const Plato::Primal       & aPrimal)
    const
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
                (tDomain, aControls, aPrimal, mLocalOrdinalMaps, tInputWorkSets);

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
                (tNumCells, aControls, aPrimal, mLocalOrdinalMaps, tInputWorkSets);

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

    /**************************************************************************//**
    * \fn Teuchos::RCP<Plato::CrsMatrixType> gradientPreviousPress
    * \brief Return gradient of residual with respet to (wrt) previous pressure (i.e. Jacobian).
    * \param [in] aControls control variables
    * \param [in] aPrimal   primal state database
    * \return Jacobian wrt previous pressure
    ******************************************************************************/
    Teuchos::RCP<Plato::CrsMatrixType>
    gradientPreviousPress
    (const Plato::ScalarVector & aControls,
     const Plato::Primal       & aPrimal)
    const
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
                (tDomain, aControls, aPrimal, mLocalOrdinalMaps, tInputWorkSets);

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
                (tNumCells, aControls, aPrimal, mLocalOrdinalMaps, tInputWorkSets);

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

    /**************************************************************************//**
    * \fn Teuchos::RCP<Plato::CrsMatrixType> gradientPreviousTemp
    * \brief Return gradient of residual with respet to (wrt) previous temperature (i.e. Jacobian).
    * \param [in] aControls control variables
    * \param [in] aPrimal   primal state database
    * \return Jacobian wrt previous temperature
    ******************************************************************************/
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

    /**************************************************************************//**
    * \fn Teuchos::RCP<Plato::CrsMatrixType> gradientCurrentVel
    * \brief Return gradient of residual with respet to (wrt) current velocity (i.e. Jacobian).
    * \param [in] aControls control variables
    * \param [in] aPrimal   primal state database
    * \return Jacobian wrt current velocity
    ******************************************************************************/
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

    /**************************************************************************//**
    * \fn Teuchos::RCP<Plato::CrsMatrixType> gradientCurrentPress
    * \brief Return gradient of residual with respet to (wrt) current pressure (i.e. Jacobian).
    * \param [in] aControls control variables
    * \param [in] aPrimal   primal state database
    * \return Jacobian wrt current pressure
    ******************************************************************************/
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

    /**************************************************************************//**
    * \fn Teuchos::RCP<Plato::CrsMatrixType> gradientCurrentTemp
    * \brief Return gradient of residual with respet to (wrt) current temperature (i.e. Jacobian).
    * \param [in] aControls control variables
    * \param [in] aPrimal   primal state database
    * \return Jacobian wrt current temperature
    ******************************************************************************/
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
    /**************************************************************************//**
    * \brief Initialize member metadata.
    * \param [in] aTag     vector function tag/type
    * \param [in] aDataMap output database
    * \param [in] aInputs  input file metadata
    ******************************************************************************/
    void initialize
    (const std::string      & aTag,
     Plato::DataMap         & aDataMap,
     Teuchos::ParameterList & aInputs)
    {
        typename PhysicsT::FunctionFactory tVecFuncFactory;
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tName = tDomain.getDomainName();

            mResidualFuncs[tName]  = tVecFuncFactory.template createVectorFunction<PhysicsT, ResidualEvalT>
                (aTag, tDomain, aDataMap, aInputs);

            mGradControlFuncs[tName] = tVecFuncFactory.template createVectorFunction<PhysicsT, GradControlEvalT>
                (aTag, tDomain, aDataMap, aInputs);

            mGradConfigFuncs[tName] = tVecFuncFactory.template createVectorFunction<PhysicsT, GradConfigEvalT>
                (aTag, tDomain, aDataMap, aInputs);

            mGradCurPressFuncs[tName] = tVecFuncFactory.template createVectorFunction<PhysicsT, GradCurPressEvalT>
                (aTag, tDomain, aDataMap, aInputs);

            mGradPrevPressFuncs[tName] = tVecFuncFactory.template createVectorFunction<PhysicsT, GradPrevPressEvalT>
                (aTag, tDomain, aDataMap, aInputs);

            mGradCurTempFuncs[tName] = tVecFuncFactory.template createVectorFunction<PhysicsT, GradCurTempEvalT>
                (aTag, tDomain, aDataMap, aInputs);

            mGradPrevTempFuncs[tName] = tVecFuncFactory.template createVectorFunction<PhysicsT, GradPrevTempEvalT>
                (aTag, tDomain, aDataMap, aInputs);

            mGradCurVelFuncs[tName] = tVecFuncFactory.template createVectorFunction<PhysicsT, GradCurVelEvalT>
                (aTag, tDomain, aDataMap, aInputs);

            mGradPrevVelFuncs[tName] = tVecFuncFactory.template createVectorFunction<PhysicsT, GradPrevVelEvalT>
                (aTag, tDomain, aDataMap, aInputs);

            mGradPredictorFuncs[tName] = tVecFuncFactory.template createVectorFunction<PhysicsT, GradPredictorEvalT>
                (aTag, tDomain, aDataMap, aInputs);
        }
    }
};
// class VectorFunction


/**************************************************************************//**
* \struct Vector and scalar function factory.
*
* \brief Responsible for the construction of vector and scalar functions.
******************************************************************************/
struct FunctionFactory
{
public:
    /**************************************************************************//**
    * \fn shared_ptr<AbstractVectorFunction> createVectorFunction
    * \tparam PhysicsT    physics type
    * \tparam EvaluationT Forward Automatic Differentiation evaluation type
    *
    * \brief Responsible for the construction of vector functions.
    *
    * \param [in] aTag     vector function tag/type
    * \param [in] aModel   struct that contains the mesh, meshsets, domains, etc.
    * \param [in] aDataMap output database
    * \param [in] aInputs  input file metadata
    *
    * \return shared pointer to an abtract vector function
    ******************************************************************************/
    template
    <typename PhysicsT,
     typename EvaluationT>
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
            return ( std::make_shared<Plato::Fluids::TemperatureResidual<PhysicsT, EvaluationT>>(aDomain, aDataMap, aInputs) );
        }
        else if( tLowerTag == "velocity predictor" )
        {
            return ( std::make_shared<Plato::Fluids::VelocityPredictorResidual<PhysicsT, EvaluationT>>(aDomain, aDataMap, aInputs) );
        }
        else
        {
            THROWERR(std::string("Vector function with tag '") + aTag + "' is not supported.")
        }
    }

    /**************************************************************************//**
    * \fn shared_ptr<AbstractScalarFunction> createScalarFunction
    * \tparam PhysicsT    physics type
    * \tparam EvaluationT Forward Automatic Differentiation evaluation type
    *
    * \brief Responsible for the construction of vector functions.
    *
    * \param [in] aTag     vector function tag/type
    * \param [in] aModel   struct that contains the mesh, meshsets, domains, etc.
    * \param [in] aDataMap output database
    * \param [in] aInputs  input file metadata
    *
    * \return shared pointer to an abtract scalar function
    ******************************************************************************/
    template
    <typename PhysicsT,
     typename EvaluationT>
    std::shared_ptr<Plato::Fluids::AbstractScalarFunction<PhysicsT, EvaluationT>>
    createScalarFunction
    (const std::string          & aTag,
     const Plato::SpatialDomain & aDomain,
     Plato::DataMap             & aDataMap,
     Teuchos::ParameterList     & aInputs)
    {
        if( !aInputs.isSublist("Criteria") )
        {
            THROWERR("'Criteria' block is not defined.")
        }
        auto tCriteriaList = aInputs.sublist("Criteria");
        if( !tCriteriaList.isSublist(aTag) )
        {
            THROWERR(std::string("Criteria Block with name '") + aTag + "' is not defined.")
        }
        auto tCriterion = tCriteriaList.sublist(aTag);

        if(!tCriterion.isParameter("Scalar Function Type"))
        {
            THROWERR(std::string("'Scalar Function Type' keyword is not defined in Criterion with name '") + aTag + "'.")
        }

        auto tFlowTag = tCriterion.get<std::string>("Flow", "Not Defined");
        auto tFlowLowerTag = Plato::tolower(tFlowTag);
        auto tCriterionTag = tCriterion.get<std::string>("Scalar Function Type", "Not Defined");
        auto tCriterionLowerTag = Plato::tolower(tCriterionTag);

        if( tCriterionLowerTag == "average surface pressure" )
        {
            return ( std::make_shared<Plato::Fluids::AverageSurfacePressure<PhysicsT, EvaluationT>>
                (aTag, aDomain, aDataMap, aInputs) );
        }
        else if( tCriterionLowerTag == "average surface temperature" )
        {
            return ( std::make_shared<Plato::Fluids::AverageSurfaceTemperature<PhysicsT, EvaluationT>>
                (aTag, aDomain, aDataMap, aInputs) );
        }
	/*
        else if( tCriterionLowerTag == "internal dissipation energy" && tFlowLowerTag == "incompressible")
        {
            return ( std::make_shared<Plato::Fluids::InternalDissipationEnergy<PhysicsT, EvaluationT>>
                (aName, aDomain, aDataMap, aInputs) );
        }
	*/
        else
        {
            THROWERR(std::string("'Scalar Function Type' with tag '") + tCriterionTag
                + "' in Criterion Block '" + aTag + "' is not supported.")
        }
    }
};
// struct FunctionFactory




/**************************************************************************//**
* \struct CriterionFactory
*
* \brief Responsible for the construction of Plato criteria.
******************************************************************************/
template<typename PhysicsT>
class CriterionFactory
{
private:
    using ScalarFunctionType = std::shared_ptr<Plato::Fluids::CriterionBase>; /*!< local scalar function type */

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
     * \brief Create criterion interface.
     * \param [in] aModel   computational model metadata
     * \param [in] aDataMap output database
     * \param [in] aInputs  input file metadata
     * \param [in] aTag    scalar function tag
     **********************************************************************************/
    ScalarFunctionType
    createCriterion
    (Plato::SpatialModel    & aModel,
     Plato::DataMap         & aDataMap,
     Teuchos::ParameterList & aInputs,
     std::string            & aTag)
     {
        auto tFunctionTag = aInputs.sublist("Criteria").sublist(aTag);
        auto tType = tFunctionTag.get<std::string>("Type", "Not Defined");
        auto tLowerType = Plato::tolower(tType);

        if(tLowerType == "scalar function")
        {
            auto tCriterion =
                std::make_shared<Plato::Fluids::ScalarFunction<PhysicsT>>
                    (aModel, aDataMap, aInputs, aTag);
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
            THROWERR(std::string("Scalar function in block '") + aTag + "' with Type '" + tType + "' is not supported.")
        }
     }
};
// class CriterionFactory


/**************************************************************************//**
* \struct WeightedScalarFunction
*
* \brief Responsible for the evaluation of a weighted scalar function.
*
* \f[
*   W(u(z),z) = \sum_{i=1}^{N_{f}}\alpha_i f_i(u(z),z)
* \f]
*
* where \f$\alpha_i\f$ is the i-th weight, \f$ f_i \f$ is the i-th scalar function,
* \f$ u(z) \f$ denotes the states, \f$ z \f$ denotes controls and \f$ N_f \f$ is
* the total number of scalar functions.
******************************************************************************/
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
    using Criterion = std::shared_ptr<Plato::Fluids::CriterionBase>; /*!< local criterion type */

    bool mDiagnostics = false; /*!< write diagnostics to terminal */
    Plato::DataMap& mDataMap; /*!< output database */
    const Plato::SpatialModel& mSpatialModel; /*!< mesh database */

    std::string mFuncTag; /*!< weighted scalar function tag */
    std::vector<Criterion>     mCriteria; /*!< list of scalar function criteria */
    std::vector<std::string>   mCriterionNames; /*!< list of criterion tags/names */
    std::vector<Plato::Scalar> mCriterionWeights; /*!< list of criterion weights */

public:
    /******************************************************************************//**
     * \brief Constructor
     * \param [in] aModel   computational model metadata
     * \param [in] aDataMap output database
     * \param [in] aInputs  input file metadata
     * \param [in] aTag    scalar function tag
     **********************************************************************************/
    WeightedScalarFunction
    (const Plato::SpatialModel    & aModel,
           Plato::DataMap         & aDataMap,
           Teuchos::ParameterList & aInputs,
           std::string            & aTag) :
         mDataMap(aDataMap),
         mSpatialModel(aModel),
         mFuncTag(aTag)
    {
        this->initialize(aInputs);
    }

    /******************************************************************************//**
     * \brief Destructor
     **********************************************************************************/
    virtual ~WeightedScalarFunction(){}

    /******************************************************************************//**
     * \brief Append scalar criterion to list.
     * \param [in] aFunc   scalar criterion
     * \param [in] aTag    scalar criterion tag/name
     * \param [in] aWeight scalar criterion weight (default = 1.0)
     **********************************************************************************/
    void append
    (const Criterion     & aFunc,
     const std::string   & aTag,
           Plato::Scalar   aWeight = 1.0)
    {
        mCriteria.push_back(aFunc);
        mCriterionNames.push_back(aTag);
        mCriterionWeights.push_back(aWeight);
    }

    /******************************************************************************//**
     * \fn std::string name
     * \brief Return scalar criterion name/tag.
     * \return scalar criterion name/tag
     **********************************************************************************/
    std::string name() const override
    {
        return mFuncTag;
    }

    /******************************************************************************//**
     * \fn Plato::Scalar value
     * \brief Evaluate scalar function.
     * \param [in] aControls control variables
     * \param [in] aPrimal   primal state database
     * \return scalar criterion value
     **********************************************************************************/
    Plato::Scalar value
    (const Plato::ScalarVector & aControls,
     const Plato::Primal       & aPrimal)
    const override
    {
        Plato::Scalar tResult = 0.0;
        for (auto& tCriterion : mCriteria)
        {
            auto tIndex = &tCriterion - &mCriteria[0];
            const auto tValue = tCriterion->value(aControls, aPrimal);
            const auto tFuncWeight = mCriterionWeights[tIndex];
            const auto tFuncValue = tFuncWeight * tValue;
            tResult += tFuncValue;

            const auto tFuncName = mCriterionNames[tIndex];
            mDataMap.mScalarValues[tFuncName] = tFuncValue;

            if(mDiagnostics)
            {
                printf("Scalar Function Name = %s \t Value = %f\n", tFuncName.c_str(), tFuncValue);
            }
        }

        if(mDiagnostics)
        {
            printf("Weighted Sum Name = %s \t Value = %f\n", mFuncTag.c_str(), tResult);
        }
        return tResult;
    }

    /******************************************************************************//**
     * \fn Plato::ScalarVector gradientConfig
     * \brief Evaluate scalar function gradient with respect to configuration (Jacobian).
     * \param [in] aControls control variables
     * \param [in] aPrimal   primal state database
     * \return Jacobian with respect to configuration
     **********************************************************************************/
    Plato::ScalarVector gradientConfig
    (const Plato::ScalarVector & aControls,
     const Plato::Primal       & aPrimal)
    const override
    {
        const auto tNumNodes = mSpatialModel.Mesh.nverts();
        Plato::ScalarVector tTotalDerivative("total derivative", mNumSpatialDims * tNumNodes);
        for (auto& tCriterion : mCriteria)
        {
            auto tIndex = &tCriterion - &mCriteria[0];
            auto tGradient = tCriterion->gradientConfig(aControls, aPrimal);
            const auto tFuncWeight = mCriterionWeights[tIndex];
            Plato::blas1::update(tFuncWeight, tGradient, static_cast<Plato::Scalar>(1.0), tTotalDerivative);
        }
        return tTotalDerivative;
    }

    /******************************************************************************//**
     * \fn Plato::ScalarVector gradientControl
     * \brief Evaluate scalar function gradient with respect to control (Jacobian).
     * \param [in] aControls control variables
     * \param [in] aPrimal   primal state database
     * \return Jacobian with respect to control
     **********************************************************************************/
    Plato::ScalarVector gradientControl
    (const Plato::ScalarVector & aControls,
     const Plato::Primal & aVariables)
    const override
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

    /******************************************************************************//**
     * \fn Plato::ScalarVector gradientCurrentPress
     * \brief Evaluate scalar function gradient with respect to current pressure (Jacobian).
     * \param [in] aControls control variables
     * \param [in] aPrimal   primal state database
     * \return Jacobian with respect to current pressure
     **********************************************************************************/
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

    /******************************************************************************//**
     * \fn Plato::ScalarVector gradientCurrentTemp
     * \brief Evaluate scalar function gradient with respect to current temperature (Jacobian).
     * \param [in] aControls control variables
     * \param [in] aPrimal   primal state database
     * \return Jacobian with respect to current temperature
     **********************************************************************************/
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

    /******************************************************************************//**
     * \fn Plato::ScalarVector gradientCurrentVel
     * \brief Evaluate scalar function gradient with respect to current velocity (Jacobian).
     * \param [in] aControls control variables
     * \param [in] aPrimal   primal state database
     * \return Jacobian with respect to current velocity
     **********************************************************************************/
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
    /******************************************************************************//**
     * \fn void checkInputs
     * \brief Check the total number of required criterion inputs match the number of functions
     **********************************************************************************/
    void checkInputs()
    {
        if (mCriterionNames.size() != mCriterionWeights.size())
        {
            THROWERR(std::string("Dimensions mismatch.  Number of 'Functions' and 'Weights' do not match. ") +
                     "Check scalar function with name '" + mFuncTag + "'.")
        }
    }

    /******************************************************************************//**
     * \fn void initialize
     * \brief Initialize member metadata
     * \param [in] aInputs  input file metadata
     **********************************************************************************/
    void initialize(Teuchos::ParameterList & aInputs)
    {
        if(aInputs.sublist("Criteria").isSublist(mFuncTag) == false)
        {
            THROWERR(std::string("Scalar function with tag '") + mFuncTag + "' is not defined in the input file.")
        }

        auto tCriteriaInputs = aInputs.sublist("Criteria").sublist(mFuncTag);
        this->parseTags(tCriteriaInputs);
        this->parseWeights(tCriteriaInputs);
        this->checkInputs();

        Plato::Fluids::CriterionFactory<PhysicsT> tFactory;
        for(auto& tName : mCriterionNames)
        {
            auto tScalarFunction = tFactory.createCriterion(mSpatialModel, mDataMap, aInputs, tName);
            mCriteria.push_back(tScalarFunction);
        }
    }

    /******************************************************************************//**
     * \fn void parseFunction
     * \brief Parse scalar function tags
     * \param [in] aInputs  input file metadata
     **********************************************************************************/
    void parseTags(Teuchos::ParameterList & aInputs)
    {
        mCriterionNames = Plato::teuchos::parse_array<std::string>("Functions", aInputs);
        if(mCriterionNames.empty())
        {
            THROWERR(std::string("'Functions' keyword was not defined in function block with name '") + mFuncTag
                + "'. User must define the 'Functions' keyword to use the 'Weighted Sum' criterion.")
        }
    }

    /******************************************************************************//**
     * \fn void parseWeights
     * \brief Parse scalar function weights
     * \param [in] aInputs  input file metadata
     **********************************************************************************/
    void parseWeights(Teuchos::ParameterList & aInputs)
    {
        mCriterionWeights = Plato::teuchos::parse_array<Plato::Scalar>("Weights", aInputs);
        if(mCriterionWeights.empty())
        {
            if(mCriterionNames.empty())
            {
                THROWERR(std::string("Criterion names were not parsed. ")
                    + "Users must define the 'Functions' keyword to use the 'Weighted Sum' criterion.")
            }
            mCriterionWeights.resize(mCriterionNames.size());
            std::fill(mCriterionWeights.begin(), mCriterionWeights.end(), 1.0);
        }
    }
};
// class WeightedScalarFunction





/**************************************************************************//**
* \struct WeightedScalarFunction
*
* \brief Responsible for the evaluation of a least squared scalar function.
*
* \f[
*   W(u(z),z) = \sum_{i=1}^{N_{f}}\alpha_i f_i(u(z),z)
* \f]
*
* where \f$\alpha_i\f$ is the i-th weight, \f$ f_i \f$ is the i-th scalar function,
* \f$ u(z) \f$ denotes the states, \f$ z \f$ denotes controls and \f$ N_f \f$ is
* the total number of scalar functions.
******************************************************************************/
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
    using Criterion = std::shared_ptr<Plato::Fluids::CriterionBase>; /*!< local criterion type */

    bool mDiagnostics; /*!< write diagnostics to terminal */
    Plato::DataMap& mDataMap; /*!< output database */
    const Plato::SpatialModel& mSpatialModel; /*!< mesh database */

    std::string mFuncName; /*!< weighted scalar function name */

    std::vector<Criterion>     mCriteria; /*!< list of scalar function criteria */
    std::vector<std::string>   mCriterionNames; /*!< list of criterion names */
    std::vector<Plato::Scalar> mCriterionTarget; /*!< list of criterion gold/target values */
    std::vector<Plato::Scalar> mCriterionWeights; /*!< list of criterion weights */
    std::vector<Plato::Scalar> mCriterionNormalizations; /*!< list of criterion normalization */

public:
    /******************************************************************************//**
     * \brief Constructor
     * \param [in] aModel   computational model metadata
     * \param [in] aDataMap output database
     * \param [in] aInputs  input file metadata
     * \param [in] aTag    scalar function tag
     **********************************************************************************/
    LeastSquaresScalarFunction
    (const Plato::SpatialModel    & aModel,
           Plato::DataMap         & aDataMap,
           Teuchos::ParameterList & aInputs,
           std::string            & aTag) :
        mDiagnostics(false),
        mDataMap(aDataMap),
        mSpatialModel(aModel),
        mFuncName(aTag)
    {
        this->initialize(aInputs);
    }

    /******************************************************************************//**
     * \fn std::string name
     * \brief Return scalar criterion name/tag.
     * \return scalar criterion name/tag
     **********************************************************************************/
    std::string name() const override
    {
        return mFuncName;
    }

    /******************************************************************************//**
     * \fn Plato::Scalar value
     * \brief Evaluate scalar function.
     * \param [in] aControls control variables
     * \param [in] aPrimal   primal state database
     * \return scalar criterion value
     **********************************************************************************/
    Plato::Scalar value
    (const Plato::ScalarVector & aControls,
     const Plato::Primal & aPrimal)
    const override
    {
        Plato::Scalar tResult = 0.0;
        for(auto& tCriterion : mCriteria)
        {
            auto tIndex = &tCriterion - &mCriteria[0];
            const auto tGold = mCriterionTarget[tIndex];
            const auto tWeight = mCriterionWeights[tIndex];
            const auto tNormalization = mCriterionNormalizations[tIndex];
            auto tCriterionValue = tCriterion->value(aControls, aPrimal);

            auto tNormalizedMisfit = (tCriterionValue - tGold) / tNormalization;
            auto tValue = tNormalizedMisfit * tNormalizedMisfit;
            tResult += tWeight * tValue * tValue;
        }
        return tResult;
    }

    /******************************************************************************//**
     * \fn Plato::ScalarVector gradientConfig
     * \brief Evaluate scalar function gradient with respect to configuration (Jacobian).
     * \param [in] aControls control variables
     * \param [in] aPrimal   primal state database
     * \return Jacobian with respect to configuration
     **********************************************************************************/
    Plato::ScalarVector gradientConfig
    (const Plato::ScalarVector & aControls,
     const Plato::Primal & aPrimal)
    const override
    {
        const auto tNumDofs = mNumConfigDofsPerNode * mSpatialModel.Mesh.nverts();
        Plato::ScalarVector tGradConfig("gradient configuration", tNumDofs);
        Plato::blas1::fill(0.0, tGradConfig);
        for(auto& tCriterion : mCriteria)
        {
            auto tIndex = &tCriterion - &mCriteria[0];
            const auto tGold = mCriterionTarget[tIndex];
            const auto tWeight = mCriterionWeights[tIndex];
            const auto tNormalization = mCriterionNormalizations[tIndex];
            auto tCriterionValue = tCriterion->value(aControls, aPrimal);
            auto tCriterionGrad  = tCriterion->gradientConfig(aControls, aPrimal);

            auto tMisfit = tCriterionValue - tGold;
            auto tMultiplier = ( static_cast<Plato::Scalar>(2.0) * tWeight * tMisfit )
                / ( tNormalization * tNormalization );
            Plato::blas1::update(tMultiplier, tCriterionGrad, 1.0, tGradConfig);
        }
        return tGradConfig;
    }

    /******************************************************************************//**
     * \fn Plato::ScalarVector gradientControl
     * \brief Evaluate scalar function gradient with respect to control (Jacobian).
     * \param [in] aControls control variables
     * \param [in] aPrimal   primal state database
     * \return Jacobian with respect to control
     **********************************************************************************/
    Plato::ScalarVector gradientControl
    (const Plato::ScalarVector & aControls,
     const Plato::Primal & aPrimal) const override
    {
        const auto tNumDofs = mNumControlDofsPerNode * mSpatialModel.Mesh.nverts();
        Plato::ScalarVector tGradControl("gradient control", tNumDofs);
        Plato::blas1::fill(0.0, tGradControl);
        for(auto& tCriterion : mCriteria)
        {
            auto tIndex = &tCriterion - &mCriteria[0];
            const auto tGold = mCriterionTarget[tIndex];
            const auto tWeight = mCriterionWeights[tIndex];
            const auto tNormalization = mCriterionNormalizations[tIndex];
            auto tCriterionValue = tCriterion->value(aControls, aPrimal);
            auto tCriterionGrad  = tCriterion->gradientControl(aControls, aPrimal);

            auto tMisfit = tCriterionValue - tGold;
            auto tMultiplier = ( static_cast<Plato::Scalar>(2.0) * tWeight * tMisfit )
                / ( tNormalization * tNormalization );
            Plato::blas1::update(tMultiplier, tCriterionGrad, 1.0, tGradControl);
        }
        return tGradControl;
    }

    /******************************************************************************//**
     * \fn Plato::ScalarVector gradientCurrentPress
     * \brief Evaluate scalar function gradient with respect to curren pressure (Jacobian).
     * \param [in] aControls control variables
     * \param [in] aPrimal   primal state database
     * \return Jacobian with respect to curren pressure
     **********************************************************************************/
    Plato::ScalarVector gradientCurrentPress
    (const Plato::ScalarVector & aControls,
     const Plato::Primal & aPrimal) const override
    {
        const auto tNumDofs = mNumPressDofsPerNode * mSpatialModel.Mesh.nverts();
        Plato::ScalarVector tGradCurPress("gradient current pressure", tNumDofs);
        Plato::blas1::fill(0.0, tGradCurPress);
        for(auto& tCriterion : mCriteria)
        {
            auto tIndex = &tCriterion - &mCriteria[0];
            const auto tGold = mCriterionTarget[tIndex];
            const auto tWeight = mCriterionWeights[tIndex];
            const auto tNormalization = mCriterionNormalizations[tIndex];
            auto tCriterionValue = tCriterion->value(aControls, aPrimal);
            auto tCriterionGrad  = tCriterion->gradientCurrentPress(aControls, aPrimal);

            auto tMisfit = tCriterionValue - tGold;
            auto tMultiplier = ( static_cast<Plato::Scalar>(2.0) * tWeight * tMisfit )
                / ( tNormalization * tNormalization );
            Plato::blas1::update(tMultiplier, tCriterionGrad, 1.0, tGradCurPress);
        }
        return tGradCurPress;
    }

    /******************************************************************************//**
     * \fn Plato::ScalarVector gradientCurrentTemp
     * \brief Evaluate scalar function gradient with respect to curren temperature (Jacobian).
     * \param [in] aControls control variables
     * \param [in] aPrimal   primal state database
     * \return Jacobian with respect to curren temperature
     **********************************************************************************/
    Plato::ScalarVector gradientCurrentTemp
    (const Plato::ScalarVector & aControls,
     const Plato::Primal & aPrimal) const override
    {
        const auto tNumDofs = mNumTempDofsPerNode * mSpatialModel.Mesh.nverts();
        Plato::ScalarVector tGradCurTemp("gradient current temperature", tNumDofs);
        Plato::blas1::fill(0.0, tGradCurTemp);
        for(auto& tCriterion : mCriteria)
        {
            auto tIndex = &tCriterion - &mCriteria[0];
            const auto tGold = mCriterionTarget[tIndex];
            const auto tWeight = mCriterionWeights[tIndex];
            const auto tNormalization = mCriterionNormalizations[tIndex];
            auto tCriterionValue = tCriterion->value(aControls, aPrimal);
            auto tCriterionGrad  = tCriterion->gradientCurrentTemp(aControls, aPrimal);

            auto tMisfit = tCriterionValue - tGold;
            auto tMultiplier = ( static_cast<Plato::Scalar>(2.0) * tWeight * tMisfit )
                / ( tNormalization * tNormalization );
            Plato::blas1::update(tMultiplier, tCriterionGrad, 1.0, tGradCurTemp);
        }
        return tGradCurTemp;
    }

    /******************************************************************************//**
     * \fn Plato::ScalarVector gradientCurrentVel
     * \brief Evaluate scalar function gradient with respect to curren velocity (Jacobian).
     * \param [in] aControls control variables
     * \param [in] aPrimal   primal state database
     * \return Jacobian with respect to curren velocity
     **********************************************************************************/
    Plato::ScalarVector gradientCurrentVel
    (const Plato::ScalarVector & aControls,
     const Plato::Primal & aPrimal) const override
    {
        const auto tNumDofs = mNumVelDofsPerNode * mSpatialModel.Mesh.nverts();
        Plato::ScalarVector tGradCurVel("gradient current velocity", tNumDofs);
        Plato::blas1::fill(0.0, tGradCurVel);
        for(auto& tCriterion : mCriteria)
        {
            auto tIndex = &tCriterion - &mCriteria[0];
            const auto tGold = mCriterionTarget[tIndex];
            const auto tWeight = mCriterionWeights[tIndex];
            const auto tNormalization = mCriterionNormalizations[tIndex];
            auto tCriterionValue = tCriterion->value(aControls, aPrimal);
            auto tCriterionGrad  = tCriterion->gradientCurrentVel(aControls, aPrimal);

            auto tMisfit = tCriterionValue - tGold;
            auto tMultiplier = ( static_cast<Plato::Scalar>(2.0) * tWeight * tMisfit )
                / ( tNormalization * tNormalization );
            Plato::blas1::update(tMultiplier, tCriterionGrad, 1.0, tGradCurVel);
        }
        return tGradCurVel;
    }

private:
    /******************************************************************************//**
     * \fn void checkInputs
     * \brief Check the total number of required criterion inputs match the number of functions
     **********************************************************************************/
    void checkInputs()
    {
        if (mCriterionNames.size() != mCriterionWeights.size())
        {
            THROWERR(std::string("Dimensions mismatch.  Number of 'Functions' and 'Weights' do not match. ") +
                     "Check inputs for scalar function with name '" + mFuncName + "'.")
        }

        if(mCriterionNames.size() != mCriterionNormalizations.size())
        {
            THROWERR(std::string("Dimensions mismatch.  Number of 'Functions' and 'Normalizations' do not match. ") +
                     "Check inputs for scalar function with name '" + mFuncName + "'.")
        }

        if(mCriterionNames.size() != mCriterionTarget.size())
        {
            THROWERR(std::string("Dimensions mismatch.  Number of 'Functions' and 'Gold/Target' values do not match. ") +
                     "Check inputs for scalar function with name '" + mFuncName + "'.")
        }
    }

    /******************************************************************************//**
     * \fn void initialize
     * \brief Initialize member metadata
     * \param [in] aInputs  input file metadata
     **********************************************************************************/
    void initialize(Teuchos::ParameterList & aInputs)
    {
        if(aInputs.sublist("Criteria").isSublist(mFuncName) == false)
        {
            THROWERR(std::string("Scalar function with tag '") + mFuncName + "' is not defined in the input file.")
        }

        auto tCriteriaInputs = aInputs.sublist("Criteria").sublist(mFuncName);
        this->parseNames(tCriteriaInputs);
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

    /******************************************************************************//**
     * \fn void parseTags
     * \brief Parse the scalar functions defining the least squares criterion.
     * \param [in] aInputs  input file metadata
     **********************************************************************************/
    void parseNames(Teuchos::ParameterList & aInputs)
    {
        mCriterionNames = Plato::teuchos::parse_array<std::string>("Functions", aInputs);
        if(mCriterionNames.empty())
        {
            THROWERR(std::string("'Functions' keyword was not defined in function block with name '") + mFuncName
                + "'. Users must define the 'Functions' keyword to use the 'Least Squares' criterion.")
        }
    }

    /******************************************************************************//**
     * \fn void parseTargets
     * \brief Parse target scalar values.
     * \param [in] aInputs  input file metadata
     **********************************************************************************/
    void parseTargets(Teuchos::ParameterList & aInputs)
    {
        mCriterionTarget = Plato::teuchos::parse_array<std::string>("Targets", aInputs);
        if(mCriterionTarget.empty())
        {
            THROWERR(std::string("'Targets' keyword was not defined in function block with name '") + mFuncName
                + "'. User must define the 'Targets' keyword to use the 'Least Squares' criterion.")
        }
    }

    /******************************************************************************//**
     * \fn void parseWeights
     * \brief Parse scalar weights. Set weights to 1.0 if these are not provided by the user.
     * \param [in] aInputs  input file metadata
     **********************************************************************************/
    void parseWeights(Teuchos::ParameterList & aInputs)
    {
        mCriterionWeights = Plato::teuchos::parse_array<Plato::Scalar>("Weights", aInputs);
        if(mCriterionWeights.empty())
        {
            if(mCriterionNames.empty())
            {
                THROWERR("Criterion names have not been parsed. Users must define the 'Functions' keyword to use the 'Least Squares' criterion.")
            }
            mCriterionWeights.resize(mCriterionNames.size());
            std::fill(mCriterionWeights.begin(), mCriterionWeights.end(), 1.0);
        }
    }

    /******************************************************************************//**
     * \fn void parseNormalization
     * \brief Parse normalization parameters. Set normalization values to 1.0 if these
     *   are not provided by the user.
     * \param [in] aInputs  input file metadata
     **********************************************************************************/
    void parseNormalization(Teuchos::ParameterList & aInputs)
    {
        mCriterionNormalizations = Plato::teuchos::parse_array<Plato::Scalar>("Normalizations", aInputs);
        if(mCriterionNormalizations.empty())
        {
            if(mCriterionNames.empty())
            {
                THROWERR("Criterion names have not been parsed. Users must define the 'Functions' keyword to use the 'Least Squares' criterion.")
            }
            mCriterionNormalizations.resize(mCriterionNames.size());
            std::fill(mCriterionNormalizations.begin(), mCriterionNormalizations.end(), 1.0);
        }
    }
};
// class LeastSquares

}
// namespace Fluids

/******************************************************************************//**
 * \class MomentumConservation
 *
 * \tparam SpaceDim    spatial dimensions (integer)
 * \tparam NumControls number of control fields (integer) - e.g. number of design materials
 *
 * \brief Defines static parameters used to solve the momentum conservation equation.
 *
 **********************************************************************************/
template
<Plato::OrdinalType SpaceDim,
 Plato::OrdinalType NumControls = 1>
class MomentumConservation : public Plato::SimplexFluids<SpaceDim, NumControls>
{
public:
    typedef Plato::Fluids::FunctionFactory FunctionFactory; /*!< local vector/scalar function factory type */
    using SimplexT = Plato::SimplexFluids<SpaceDim, NumControls>; /*!< local simplex element type */

    static constexpr Plato::OrdinalType mNumDofsPerNode = SimplexT::mNumMomentumDofsPerNode; /*!< number of momentum degress of freedom per node */
    static constexpr Plato::OrdinalType mNumDofsPerCell = SimplexT::mNumNodesPerCell * mNumDofsPerNode; /*!< number of momentum degress of freedom per cell */
};
// class MomentumConservation

/******************************************************************************//**
 * \class MassConservation
 *
 * \tparam SpaceDim    spatial dimensions (integer)
 * \tparam NumControls number of control fields (integer) - e.g. number of design materials
 *
 * \brief Defines static parameters used to solve the mass conservation equation.
 *
 **********************************************************************************/
template
<Plato::OrdinalType SpaceDim,
 Plato::OrdinalType NumControls = 1>
class MassConservation : public Plato::SimplexFluids<SpaceDim, NumControls>
{
public:
    typedef Plato::Fluids::FunctionFactory FunctionFactory; /*!< local vector/scalar function factory type */
    using SimplexT = Plato::SimplexFluids<SpaceDim, NumControls>; /*!< local simplex element type */

    static constexpr Plato::OrdinalType mNumDofsPerNode = SimplexT::mNumMassDofsPerNode; /*!< number of mass degress of freedom per node */
    static constexpr Plato::OrdinalType mNumDofsPerCell = SimplexT::mNumNodesPerCell * mNumDofsPerNode; /*!< number of mass degress of freedom per cell */
};
// class MassConservation

/******************************************************************************//**
 * \class EnergyConservation
 *
 * \tparam SpaceDim    spatial dimensions (integer)
 * \tparam NumControls number of control fields (integer) - e.g. number of design materials
 *
 * \brief Defines static parameters used to solve the energy conservation equation.
 *
 **********************************************************************************/
template
<Plato::OrdinalType SpaceDim,
 Plato::OrdinalType NumControls = 1>
class EnergyConservation : public Plato::SimplexFluids<SpaceDim, NumControls>
{
public:
    typedef Plato::Fluids::FunctionFactory FunctionFactory; /*!< local vector/scalar function factory type */
    using SimplexT = Plato::SimplexFluids<SpaceDim, NumControls>; /*!< local simplex element type */

    static constexpr Plato::OrdinalType mNumDofsPerNode = SimplexT::mNumEnergyDofsPerNode; /*!< number of energy degress of freedom per node */
    static constexpr Plato::OrdinalType mNumDofsPerCell = SimplexT::mNumNodesPerCell * mNumDofsPerNode; /*!< number of energy degress of freedom per cell */
};
// class EnergyConservation

/******************************************************************************//**
 * \class IncompressibleFluids
 *
 * \tparam SpaceDim    spatial dimensions (integer)
 * \tparam NumControls number of control fields (integer) - e.g. number of design materials
 *
 * \brief Defines static parameters used to solve incompressible fluid flow problems.
 *
 **********************************************************************************/
template
<Plato::OrdinalType SpaceDim,
 Plato::OrdinalType NumControls = 1>
class IncompressibleFluids : public Plato::SimplexFluids<SpaceDim, NumControls>
{
public:
    static constexpr auto mNumSpatialDims = SpaceDim; /*!< number of spatial dimensions */

    typedef Plato::Fluids::FunctionFactory FunctionFactory; /*!< local vector/scalar function factory type */
    using SimplexT = typename Plato::SimplexFluids<SpaceDim, NumControls>; /*!< local simplex element type */

    using MassPhysicsT     = typename Plato::MassConservation<SpaceDim, NumControls>; /*!< local mass conservation physics type */
    using EnergyPhysicsT   = typename Plato::EnergyConservation<SpaceDim, NumControls>; /*!< local energy conservation physics type */
    using MomentumPhysicsT = typename Plato::MomentumConservation<SpaceDim, NumControls>; /*!< local momentum conservation physics type */
};
// class IncompressibleFluids



namespace cbs
{


/******************************************************************************//**
 * \fn inline Plato::ScalarVector calculate_element_characteristic_sizes
 *
 * \tparam NumSpatialDims  spatial dimensions (integer)
 * \tparam NumNodesPerCell number of nodes per cell (integer)
 *
 * \brief Calculate characteristic size for all the elements on the finite element mesh.
 *
 * \param [in] aModel spatial model database, holds such as mesh information.
 * \return array of element characteristic size
 *
 **********************************************************************************/
template
<Plato::OrdinalType NumSpatialDims,
 Plato::OrdinalType NumNodesPerCell>
inline Plato::ScalarVector
calculate_element_characteristic_sizes
(const Plato::SpatialModel & aModel)
{
    auto tCoords = aModel.Mesh.coords();
    auto tCells2Nodes = aModel.Mesh.ask_elem_verts();

    Plato::OrdinalType tNumCells = aModel.Mesh.nelems();
    Plato::OrdinalType tNumNodes = aModel.Mesh.nverts();
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
// function calculate_element_characteristic_sizes

/******************************************************************************//**
 * \fn inline Plato::ScalarVector calculate_convective_velocity_magnitude
 *
 * \tparam NodesPerCell number of nodes per cell (integer)
 *
 * \brief Calculate convective velocity magnitude at each node.
 *
 * \param [in] aModel    spatial model database, holds such as mesh information
 * \param [in] aVelocity velocity field
 *
 * \return convective velocity magnitude at each node
 *
 **********************************************************************************/
template<Plato::OrdinalType NodesPerCell>
Plato::ScalarVector
calculate_convective_velocity_magnitude
(const Plato::SpatialModel & aModel,
 const Plato::ScalarVector & aVelocity)
{
    auto tCell2Node = aModel.Mesh.ask_elem_verts();
    Plato::OrdinalType tSpaceDim = aModel.Mesh.dim();
    Plato::OrdinalType tNumCells = aModel.Mesh.nelems();
    Plato::OrdinalType tNumNodes = aModel.Mesh.nverts();

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
                tSum += aVelocity(tDofIndex) * aVelocity(tDofIndex);
            }
            auto tMyValue = sqrt(tSum);
            tConvectiveVelocity(tVertexIndex) =
                tMyValue >= tConvectiveVelocity(tVertexIndex) ? tMyValue : tConvectiveVelocity(tVertexIndex);
        }
    }, "calculate_convective_velocity_magnitude");

    return tConvectiveVelocity;
}
// function calculate_convective_velocity_magnitude

/******************************************************************************//**
 * \fn inline Plato::Scalar calculate_critical_diffusion_time_step
 *
 * \brief Calculate critical diffusion time step.
 *
 * \param [in] aKinematicViscocity kinematic viscocity
 * \param [in] aThermalDiffusivity thermal diffusivity
 * \param [in] aCharElemSize       characteristic element size
 * \param [in] aSafetyFactor       safety factor
 *
 * \return critical diffusive time step scalar
 *
 **********************************************************************************/
inline Plato::Scalar
calculate_critical_diffusion_time_step
(const Plato::Scalar aKinematicViscocity,
 const Plato::Scalar aThermalDiffusivity,
 const Plato::ScalarVector & aCharElemSize,
 Plato::Scalar aSafetyFactor = 0.7)
{
    auto tNumNodes = aCharElemSize.size();
    Plato::ScalarVector tLocalTimeStep("time step", tNumNodes);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumNodes), LAMBDA_EXPRESSION(const Plato::OrdinalType & aNodeOrdinal)
    {
        auto tKinematicStep = ( aSafetyFactor * aCharElemSize(aNodeOrdinal) * aCharElemSize(aNodeOrdinal) ) /
                ( static_cast<Plato::Scalar>(2) * aKinematicViscocity );
        auto tDiffusivityStep = ( aSafetyFactor * aCharElemSize(aNodeOrdinal) * aCharElemSize(aNodeOrdinal) ) /
                ( static_cast<Plato::Scalar>(2) * aThermalDiffusivity );
        tLocalTimeStep(aNodeOrdinal) = tKinematicStep < tDiffusivityStep ? tKinematicStep : tDiffusivityStep;
    }, "calculate local critical time step");

    Plato::Scalar tMinValue = 0.0;
    Plato::blas1::min(tLocalTimeStep, tMinValue);
    return tMinValue;
}
// function calculate_critical_diffusion_time_step

/******************************************************************************//**
 * \fn inline Plato::Scalar calculate_critical_time_step_upper_bound
 *
 * \brief Calculate critical time step upper bound.
 *
 * \param [in] aVelUpperBound critical velocity lower bound
 * \param [in] aCharElemSize  characteristic element size
 *
 * \return critical time step upper bound (scalar)
 *
 **********************************************************************************/
inline Plato::Scalar 
calculate_critical_time_step_upper_bound
(const Plato::Scalar aVelUpperBound,
 const Plato::ScalarVector& aCharElemSize)
{
    Plato::Scalar tMinValue = 0.0;
    Plato::blas1::min(aCharElemSize, tMinValue);
    auto tOutput = tMinValue / aVelUpperBound;
    return tOutput;
}
// function calculate_critical_time_step_upper_bound


/******************************************************************************//**
 * \fn inline Plato::Scalar calculate_critical_convective_time_step
 *
 * \brief Calculate critical convective time step.
 *
 * \param [in] aModel spatial model metadata
 * \param [in] aCharElemSize  characteristic element size
 * \param [in] aVelocity      velocity field
 * \param [in] aSafetyFactor  safety factor multiplier (default = 0.7)
 *
 * \return critical convective time step (scalar)
 *
 **********************************************************************************/
inline Plato::Scalar
calculate_critical_convective_time_step
(const Plato::SpatialModel & aModel,
 const Plato::ScalarVector & aCharElemSize,
 const Plato::ScalarVector & aVelocity,
 Plato::Scalar aSafetyFactor = 0.7)
{
    auto tNorm = Plato::blas1::norm(aVelocity);
    if(tNorm <= std::numeric_limits<Plato::Scalar>::min())
    {
        return std::numeric_limits<Plato::Scalar>::max();
    }

    auto tNumNodes = aModel.Mesh.nverts();
    Plato::ScalarVector tLocalTimeStep("time step", tNumNodes);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumNodes), LAMBDA_EXPRESSION(const Plato::OrdinalType & aNodeOrdinal)
    {
        tLocalTimeStep(aNodeOrdinal) = aSafetyFactor * ( aCharElemSize(aNodeOrdinal) / aVelocity(aNodeOrdinal) );
    }, "calculate local critical time step");

    Plato::Scalar tMinValue = 0;
    Plato::blas1::min(tLocalTimeStep, tMinValue);
    return tMinValue;
}
// function calculate_critical_convective_time_step

/******************************************************************************//**
 * \fn inline void enforce_boundary_condition
 *
 * \brief Enforce boundary conditions.
 *
 * \param [in] aBcDofs    degrees of freedom associated with the boundary conditions
 * \param [in] aBcValues  values enforced in boundary degrees of freedom
 * \param [in/out] aState physical field
 *
 **********************************************************************************/
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
// function enforce_boundary_condition

/******************************************************************************//**
 * \fn inline Plato::ScalarVector calculate_field_misfit
 *
 * \tparam DofsPerNode degrees of freedom per node (integer)
 *
 * \brief Calculate misfit between two fields per degree of freedom.
 *
 * \param [in] aNumNodes number of nodes in the mesh
 * \param [in] aFieldOne physical field one
 * \param [in] aFieldTwo physical field two
 *
 * \return misfit per degree of freedom
 *
 **********************************************************************************/
template<Plato::OrdinalType DofsPerNode>
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
    }, "calculate field misfit");

    return tResidual;
}
// function calculate_field_misfit

/******************************************************************************//**
 * \fn inline Plato::Scalar calculate_misfit_euclidean_norm
 *
 * \tparam DofsPerNode degrees of freedom per node (integer)
 *
 * \brief Calculate euclidean norm of the misfit between two fields.
 *
 * \param [in] aNumNodes number of nodes in the mesh
 * \param [in] aFieldOne physical field one
 * \param [in] aFieldTwo physical field two
 *
 * \return euclidean norm scalar
 *
 **********************************************************************************/
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
// function calculate_misfit_euclidean_norm


/******************************************************************************//**
 * \fn inline Plato::Scalar calculate_misfit_inf_norm
 *
 * \tparam DofsPerNode degrees of freedom per node (integer)
 *
 * \brief Calculate infinite norm of the misfit between two fields.
 *
 * \param [in] aNumNodes number of nodes in the mesh
 * \param [in] aFieldOne physical field one
 * \param [in] aFieldTwo physical field two
 *
 * \return euclidean norm scalar
 *
 **********************************************************************************/
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
// function calculate_misfit_inf_norm

}
// namespace cbs


/******************************************************************************//**
 * \fn inline void apply_constraints
 *
 * \tparam DofsPerNode degrees of freedom per node (integer)
 *
 * \brief Apply constraints to system of equations by modifying left and right hand sides.
 *
 * \param [in]     aBcDofs   degrees of freedom (dofs) associated with the boundary conditions
 * \param [in]     aBcValues scalar values forced at the dofs where the boundary conditions are applied
 * \param [in]     aScale    scalar multiplier
 * \param [in/out] aMatrix   left-hand-side matrix
 * \param [in/out] aRhs      right-hand-side vector
 *
 **********************************************************************************/
template<Plato::OrdinalType DofsPerNode>
inline void apply_constraints
(const Plato::LocalOrdinalVector          & aBcDofs,
 const Plato::ScalarVector                & aBcValues,
 const Teuchos::RCP<Plato::CrsMatrixType> & aMatrix,
       Plato::ScalarVector                & aRhs,
       Plato::Scalar                        aScale = 1.0)
{
    if(aMatrix->isBlockMatrix())
    {
        Plato::applyBlockConstraints<DofsPerNode>(aMatrix, aRhs, aBcDofs, aBcValues, aScale);
    }
    else
    {
        Plato::applyConstraints<DofsPerNode>(aMatrix, aRhs, aBcDofs, aBcValues, aScale);
    }
}
// function apply_constraints

/******************************************************************************//**
 * \fn inline void set_dofs_values
 *
 * \brief Set values at degrees of freedom to input scalar (default scalar = 0.0).
 *
 * \param [in]     aBcDofs list of degrees of freedom (dofs)
 * \param [in]     aValue  scalar value (default = 0.0)
 * \param [in/out] aOutput output vector
 *
 **********************************************************************************/
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
// function set_dofs_values

/******************************************************************************//**
 * \fn inline void open_text_file
 *
 * \brief Open text file.
 *
 * \param [in]     aFileName filename
 * \param [in]     aPrint    boolean flag (true = open file, false = do not open)
 * \param [in/out] aTextFile text file macro
 *
 **********************************************************************************/
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
// function open_text_file

/******************************************************************************//**
 * \fn inline void close_text_file
 *
 * \brief Close text file.
 *
 * \param [in]     aPrint    boolean flag (true = close file, false = do not close)
 * \param [in/out] aTextFile text file macro
 *
 **********************************************************************************/
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
// function close_text_file

/******************************************************************************//**
 * \fn inline void append_text_to_file
 *
 * \brief Append text to file.
 *
 * \param [in]     aMsg      text message to be appended to file
 * \param [in]     aPrint    boolean flag (true = print message to file, false = do not print message)
 * \param [in/out] aTextFile text file macro
 *
 **********************************************************************************/
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
// function append_text_to_file

namespace Fluids
{

/******************************************************************************//**
 * \class AbstractProblem
 *
 * \brief This pure virtual class provides blueprint for any derived class.
 *   Derived classes define the main interface used to solve a Plato problem.
 *
 **********************************************************************************/
class AbstractProblem
{
public:
    virtual ~AbstractProblem() {}

    /******************************************************************************//**
     * \fn void output
     *
     * \brief Output interface to permit output of quantities of interests to a visualization file.
     *
     * \param [in] aFilePath visualization file path
     *
     **********************************************************************************/
    virtual void output(std::string aFilePath) = 0;

    /******************************************************************************//**
     * \fn const Plato::DataMap& getDataMap
     *
     * \brief Return a constant reference to the Plato output database.
     * \return constant reference to the Plato output database
     *
     **********************************************************************************/
    virtual const Plato::DataMap& getDataMap() const = 0;

    /******************************************************************************//**
     * \fn Plato::Solutions solution
     *
     * \brief Solve finite element simulation.
     * \param [in] aControl vector of design/optimization variables
     * \return Plato database with state solutions
     *
     **********************************************************************************/
    virtual Plato::Solutions solution(const Plato::ScalarVector& aControl) = 0;

    /******************************************************************************//**
     * \fn Plato::Scalar criterionValue
     *
     * \brief Evaluate criterion.
     * \param [in] aControl vector of design/optimization variables
     * \param [in] aName    criterion name/identifier
     * \return criterion evaluation
     *
     **********************************************************************************/
    virtual Plato::Scalar criterionValue(const Plato::ScalarVector & aControl, const std::string& aName) = 0;

    /******************************************************************************//**
     * \fn Plato::Scalar criterionGradient
     *
     * \brief Evaluate criterion gradient with respect to design/optimization variables.
     * \param [in] aControl vector of design/optimization variables
     * \param [in] aName    criterion name/identifier
     * \return criterion gradient with respect to design/optimization variables
     *
     **********************************************************************************/
    virtual Plato::ScalarVector criterionGradient(const Plato::ScalarVector &aControl, const std::string &aName) = 0;

    /******************************************************************************//**
     * \fn Plato::Scalar criterionGradientX
     *
     * \brief Evaluate criterion gradient with respect to configuration variables.
     * \param [in] aControl vector of design/optimization variables
     * \param [in] aName    criterion name/identifier
     * \return criterion gradient with respect to configuration variables
     *
     **********************************************************************************/
    virtual Plato::ScalarVector criterionGradientX(const Plato::ScalarVector &aControl, const std::string &aName) = 0;
};
// class AbstractProblem

/******************************************************************************//**
 * \class QuasiImplicit
 *
 * \brief Main interface for the steady-state solution of incompressible fluid flow problems.
 *
 **********************************************************************************/
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
    const Teuchos::ParameterList& mInputs; /*!< input file metadata */

    Plato::DataMap mDataMap; /*!< static output fields metadata interface */
    Plato::SpatialModel mSpatialModel; /*!< SpatialModel instance contains the mesh, meshsets, domains, etc. */

    bool mPrintDiagnostics = true; /*!< boolean flag use to output solver diagnostics to file */
    bool mCalculateHeatTransfer = false; /*!< boolean flag use to enable heat transfer calculations */

    std::ofstream mDiagnostics; /*!< output diagnostics */

    Plato::Scalar mTimeStepDamping = 1.0; /*!< time step damping */
    Plato::Scalar mPressureTolerance = 1e-4; /*!< pressure solver stopping tolerance */
    Plato::Scalar mPredictorTolerance = 1e-4; /*!< velocity predictor solver stopping tolerance */
    Plato::Scalar mCorrectorTolerance = 1e-4; /*!< velocity corrector solver stopping tolerance */
    Plato::Scalar mTemperatureTolerance = 1e-2; /*!< temperature solver stopping tolerance */
    Plato::Scalar mSteadyStateTolerance = 1e-5; /*!< steady-state stopping tolerance */
    Plato::Scalar mTimeStepSafetyFactor = 0.7; /*!< safety factor applied to stable time step */
    Plato::Scalar mCriticalThermalDiffusivity = 1.0; /*!< fluid thermal diffusivity - used to calculate stable time step */
    Plato::Scalar mCriticalKinematicViscocity = 1.0; /*!< fluid kinematic viscocity - used to calculate stable time step */
    Plato::Scalar mCriticalVelocityLowerBound = 0.5; /*!< dimensionless critical convective velocity upper bound */

    Plato::OrdinalType mOutputFrequency = 1e6; /*!< output frequency */
    Plato::OrdinalType mMaxPressureIterations = 5; /*!< maximum number of pressure solver iterations */
    Plato::OrdinalType mMaxPredictorIterations = 5; /*!< maximum number of predictor solver iterations */
    Plato::OrdinalType mMaxCorrectorIterations = 5; /*!< maximum number of corrector solver iterations */
    Plato::OrdinalType mMaxTemperatureIterations = 5; /*!< maximum number of temperature solver iterations */
    Plato::OrdinalType mNumForwardSolveTimeSteps = 0; /*!< number of time steps taken to reach steady state */
    Plato::OrdinalType mMaxSteadyStateIterations = 1000; /*!< maximum number of steady state iterations */

    // primal state containers 
    Plato::ScalarMultiVector mPressure; /*!< pressure solution at time step n and n-1 */
    Plato::ScalarMultiVector mVelocity; /*!< velocity solution at time step n and n-1 */
    Plato::ScalarMultiVector mPredictor; /*!< velocity predictor solution at time step n and n-1 */
    Plato::ScalarMultiVector mTemperature; /*!< temperature solution at time step n and n-1 */

    // adjoint state containers
    Plato::ScalarMultiVector mAdjointPressure; /*!< adjoint pressure solution at time step n and n+1 */
    Plato::ScalarMultiVector mAdjointVelocity; /*!< adjoint velocity solution at time step n and n+1 */
    Plato::ScalarMultiVector mAdjointPredictor; /*!< adjoint velocity predictor solution at time step n and n+1 */
    Plato::ScalarMultiVector mAdjointTemperature; /*!< adjoint temperature solution at time step n and n+1 */

    // critical time step container
    std::vector<Plato::Scalar> mCriticalTimeStepHistory; /*!< critical time step history */

    // vector functions
    Plato::Fluids::VectorFunction<typename PhysicsT::MassPhysicsT>     mPressureResidual; /*!< pressure solver vector function interface */
    Plato::Fluids::VectorFunction<typename PhysicsT::MomentumPhysicsT> mPredictorResidual; /*!< velocity predictor solver vector function interface */
    Plato::Fluids::VectorFunction<typename PhysicsT::MomentumPhysicsT> mCorrectorResidual; /*!< velocity corrector solver vector function interface */
    // Using pointer since default VectorFunction constructor allocations are not permitted.
    // Temperature VectorFunction allocation is optional since heat transfer calculations are optional
    std::shared_ptr<Plato::Fluids::VectorFunction<typename PhysicsT::EnergyPhysicsT>> mTemperatureResidual; /*!< temperature solver vector function interface */

    // optimization problem criteria
    using Criterion = std::shared_ptr<Plato::Fluids::CriterionBase>; /*!< local criterion type */
    using Criteria  = std::unordered_map<std::string, Criterion>; /*!< local criterion list type */
    Criteria mCriteria;  /*!< criteria list */

    // local conservation equation, i.e. physics, types
    using MassConservationT     = typename Plato::MassConservation<PhysicsT::mNumSpatialDims, PhysicsT::mNumControlDofsPerNode>; /*!< local mass conservation equation type */
    using EnergyConservationT   = typename Plato::EnergyConservation<PhysicsT::mNumSpatialDims, PhysicsT::mNumControlDofsPerNode>; /*!< local energy conservation equation type */
    using MomentumConservationT = typename Plato::MomentumConservation<PhysicsT::mNumSpatialDims, PhysicsT::mNumControlDofsPerNode>; /*!< local momentum conservation equation type */

    // essential boundary conditions accessors
    Plato::EssentialBCs<MassConservationT>     mPressureEssentialBCs; /*!< pressure essential/Dirichlet boundary condition interface */
    Plato::EssentialBCs<MomentumConservationT> mVelocityEssentialBCs; /*!< velocity essential/Dirichlet boundary condition interface */
    Plato::EssentialBCs<EnergyConservationT>   mTemperatureEssentialBCs; /*!< temperature essential/Dirichlet boundary condition interface */

public:
    /******************************************************************************//**
     * \brief Constructor
     * \param [in] aMesh     finite element mesh metadata
     * \param [in] aMeshSets mesh entity sets metadata
     * \param [in] aInputs   input file metadata
     * \param [in] aMachine  input file metadata
     **********************************************************************************/
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

    /******************************************************************************//**
     * \brief Destructor
     **********************************************************************************/
    ~QuasiImplicit()
    {
        if(Plato::Comm::rank(mMachine) == 0)
        {
            Plato::close_text_file(mDiagnostics, mPrintDiagnostics);
        }
    }

    /******************************************************************************//**
     * \fn const Plato::DataMap getDataMap
     * \brief Return constant reference to Plato output database.
     * \return constant reference to Plato output database
     **********************************************************************************/
    const decltype(mDataMap)& getDataMap() const
    {
        return mDataMap;
    }

    /******************************************************************************//**
     * \fn void output
     * \brief Output solution to visualization file.
     * \param [in] aFilePath visualization file path (default = ./output)
     **********************************************************************************/
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

    /******************************************************************************//**
     * \fn void write
     * \brief Write solution to visualization file. This function is mostly used for
     *   optimization purposes to avoid storing large time-dependent state history in
     *   memory. Thus, maximizing available GPU memory.
     *
     * \param [in] aPrimal primal state database
     * \param [in] aWriter interface to allow output to a VTK visualization file
     *
     **********************************************************************************/
    void write
    (const Plato::Primal& aPrimal,
     Omega_h::vtk::Writer& aWriter)
    {
        constexpr auto tStride = 0;
        const auto tNumNodes = mSpatialModel.Mesh.nverts();
        const Plato::OrdinalType tTimeStepIndex = aPrimal.scalar("time step index");

        std::string tTag = tTimeStepIndex != static_cast<Plato::OrdinalType>(0) ? "current pressure" : "previous pressure";
        auto tPressureView = aPrimal.vector(tTag);
        Omega_h::Write<Omega_h::Real> tPressure(tPressureView.size(), "Pressure");
        Plato::copy<mNumPressDofsPerNode, mNumPressDofsPerNode>(tStride, tNumNodes, tPressureView, tPressure);
        mSpatialModel.Mesh.add_tag(Omega_h::VERT, "Pressure", mNumPressDofsPerNode, Omega_h::Reals(tPressure));

        tTag = tTimeStepIndex != static_cast<Plato::OrdinalType>(0) ? "current velocity" : "previous velocity";
        auto tVelocityView = aPrimal.vector(tTag);
        Omega_h::Write<Omega_h::Real> tVelocity(tVelocityView.size(), "Velocity");
        Plato::copy<mNumVelDofsPerNode, mNumVelDofsPerNode>(tStride, tNumNodes, tVelocityView, tVelocity);
        mSpatialModel.Mesh.add_tag(Omega_h::VERT, "Velocity", mNumVelDofsPerNode, Omega_h::Reals(tVelocity));

        tTag = tTimeStepIndex != static_cast<Plato::OrdinalType>(0) ? "current predictor" : "previous predictor";
        auto tPredictorView = aPrimal.vector(tTag);
        Omega_h::Write<Omega_h::Real> tPredictor(tPredictorView.size(), "Predictor");
        Plato::copy<mNumVelDofsPerNode, mNumVelDofsPerNode>(tStride, tNumNodes, tPredictorView, tPredictor);
        mSpatialModel.Mesh.add_tag(Omega_h::VERT, "Predictor", mNumVelDofsPerNode, Omega_h::Reals(tPredictor));

        if(mCalculateHeatTransfer)
        {
            tTag = tTimeStepIndex != static_cast<Plato::OrdinalType>(0) ? "current temperature" : "previous temperature";
            auto tTemperatureView = aPrimal.vector(tTag);
            Omega_h::Write<Omega_h::Real> tTemperature(tTemperatureView.size(), "Temperature");
            Plato::copy<mNumTempDofsPerNode, mNumTempDofsPerNode>(tStride, tNumNodes, tTemperatureView, tTemperature);
            mSpatialModel.Mesh.add_tag(Omega_h::VERT, "Temperature", mNumTempDofsPerNode, Omega_h::Reals(tTemperature));
        }

        auto tTags = Omega_h::vtk::get_all_vtk_tags(&mSpatialModel.Mesh, mNumSpatialDims);
        aWriter.write(tTimeStepIndex, tTimeStepIndex, tTags);
    }

    /******************************************************************************//**
     * \fn Plato::Solutions solution
     *
     * \brief Solve finite element simulation.
     * \param [in] aControl vector of design/optimization variables
     * \return Plato database with state solutions
     *
     **********************************************************************************/
    Plato::Solutions solution
    (const Plato::ScalarVector& aControl)
    {
        this->clear();
        this->checkProblemSetup();

        Plato::Primal tPrimal;
        auto tWriter = Omega_h::vtk::Writer("solution_history", &mSpatialModel.Mesh, mNumSpatialDims);
        this->setInitialConditions(tPrimal, tWriter);
        this->calculateCharacteristicElemSize(tPrimal);

        for(Plato::OrdinalType tIteration = 0; tIteration < mMaxSteadyStateIterations; tIteration++)
        {
            mNumForwardSolveTimeSteps = tIteration + 1;
            tPrimal.scalar("time step index", mNumForwardSolveTimeSteps);

            this->setPrimal(tPrimal);
            this->calculateCriticalTimeStep(tPrimal);
            this->checkCriticalTimeStep(tPrimal);

            this->printIteration(tPrimal);
            this->updatePredictor(aControl, tPrimal);
            this->updatePressure(aControl, tPrimal);
            this->updateCorrector(aControl, tPrimal);

            if(mCalculateHeatTransfer)
            {
                this->updateTemperature(aControl, tPrimal);
            }

            if(this->writeOutput(tIteration))
            {
                this->write(tPrimal, tWriter);
            }

            if(this->checkStoppingCriteria(tPrimal))
            {
                break;
            }
            this->savePrimal(tPrimal);
        }

        auto tSolution = this->setSolution();
        return tSolution;
    }

    /******************************************************************************//**
     * \fn Plato::Scalar criterionValue
     *
     * \brief Evaluate criterion.
     * \param [in] aControl  vector of design/optimization variables
     * \param [in] aSolution Plato database with state solutions
     * \param [in] aName     criterion name/identifier
     * \return criterion evaluation
     *
     **********************************************************************************/
    Plato::Scalar criterionValue
    (const Plato::ScalarVector & aControl,
     const Plato::Solutions    & aSolution,
     const std::string         & aName)
    {
        return (this->criterionValue(aControl, aName));
    }

    /******************************************************************************//**
     * \fn Plato::Scalar criterionValue
     *
     * \brief Evaluate criterion.
     * \param [in] aControl  vector of design/optimization variables
     * \param [in] aName     criterion name/identifier
     * \return criterion evaluation
     *
     **********************************************************************************/
    Plato::Scalar criterionValue
    (const Plato::ScalarVector & aControl,
     const std::string         & aName)
    {
        auto tItr = mCriteria.find(aName);
        if (tItr == mCriteria.end())
        {
            THROWERR(std::string("Criterion with tag '") + aName + "' is not in the criteria list");
        }

        auto tDirectory = std::string("solution_history");
        auto tSolutionHistory = Plato::omega_h::read_pvtu_file_paths(tDirectory);
        if(tSolutionHistory.size() != static_cast<size_t>(mNumForwardSolveTimeSteps + 1))
        {
            THROWERR(std::string("Number of time steps read from the '") + tDirectory
                 + "' directory does not match the expected value: '" + std::to_string(mNumForwardSolveTimeSteps + 1) + "'.")
        }

        // evaluate steady-state criterion
        Plato::Primal tPrimal;
        auto tLastTimeStepIndex = tSolutionHistory.size() - 1u;
        tPrimal.scalar("time step index", tLastTimeStepIndex);
        this->setPrimal(tSolutionHistory, tPrimal);
        auto tOutput = tItr->second->value(aControl, tPrimal);

     	return tOutput;
    }

    /******************************************************************************//**
     * \fn Plato::Scalar criterionGradient
     *
     * \brief Evaluate criterion gradient with respect to design/optimization variables.
     * \param [in] aControl  vector of design/optimization variables
     * \param [in] aSolution Plato database with state solutions
     * \param [in] aName     criterion name/identifier
     * \return criterion gradient with respect to design/optimization variables
     *
     **********************************************************************************/
    Plato::ScalarVector criterionGradient
    (const Plato::ScalarVector & aControl,
     const Plato::Solutions    & aSolution,
     const std::string         & aName)
    {
        return (this->criterionGradient(aControl, aName));
    }

    /******************************************************************************//**
     * \fn Plato::Scalar criterionGradient
     *
     * \brief Evaluate criterion gradient with respect to design/optimization variables.
     * \param [in] aControl vector of design/optimization variables
     * \param [in] aName    criterion name/identifier
     * \return criterion gradient with respect to design/optimization variables
     *
     **********************************************************************************/
    Plato::ScalarVector criterionGradient
    (const Plato::ScalarVector & aControl,
     const std::string         & aName)
    {
        auto tItr = mCriteria.find(aName);
        if (tItr == mCriteria.end())
        {
            THROWERR(std::string("Criterion with tag '") + aName + "' is not defined in critria map.");
        }

        Plato::Dual tDual;
        Plato::Primal tCurrentState, tPreviousState;
        auto tDirectory = std::string("solution_history");
        auto tSolutionHistoryPaths = Plato::omega_h::read_pvtu_file_paths(tDirectory);
        if(tSolutionHistoryPaths.size() != static_cast<size_t>(mNumForwardSolveTimeSteps + 1))
        {
            THROWERR(std::string("Number of time steps read from the '") + tDirectory
                 + "' directory does not match the expected value: '" + std::to_string(mNumForwardSolveTimeSteps + 1) + "'.")
        }

        Plato::ScalarVector tTotalDerivative("total derivative", mSpatialModel.Mesh.nverts());
        for(auto tItr = tSolutionHistoryPaths.rbegin(); tItr != tSolutionHistoryPaths.rend() - 1; tItr++)
        {
            // set fields for the current primal state
            auto tCurrentStateIndex = (tSolutionHistoryPaths.size() - 1u) - std::distance(tSolutionHistoryPaths.rbegin(), tItr);
            tCurrentState.scalar("time step index", tCurrentStateIndex);
            this->setPrimal(tSolutionHistoryPaths, tCurrentState);
            this->setCriticalTimeStep(tCurrentState);

	        // set fields for the previous primal state
            auto tPreviousStateIndex = tCurrentStateIndex + 1u;
            tPreviousState.scalar("time step index", tPreviousStateIndex);
            if(tPreviousStateIndex != tSolutionHistoryPaths.size())
            {
                this->setPrimal(tSolutionHistoryPaths, tPreviousState);
                this->setCriticalTimeStep(tPreviousState);
            }

	        // set adjoint state
            this->setDual(tDual);

            // update adjoint states
            if(mCalculateHeatTransfer)
            {
                this->updateTemperatureAdjoint(aName, aControl, tCurrentState, tPreviousState, tDual);
            }
            this->updateCorrectorAdjoint(aName, aControl, tCurrentState, tPreviousState, tDual);
            this->updatePressureAdjoint(aName, aControl, tCurrentState, tPreviousState, tDual);
            this->updatePredictorAdjoint(aControl, tCurrentState, tPreviousState, tDual);

            // update total derivative with respect to control variables
            this->updateTotalDerivativeWrtControl(aName, aControl, tCurrentState, tDual, tTotalDerivative);

            this->saveDual(tDual);
        }
        return tTotalDerivative;
    }

    /******************************************************************************//**
     * \fn Plato::Scalar criterionGradientX
     *
     * \brief Evaluate criterion gradient with respect to configuration variables.
     * \param [in] aControl vector of design/optimization variables
     * \param [in] aName    criterion name/identifier
     * \return criterion gradient with respect to configuration variables
     *
     **********************************************************************************/
    Plato::ScalarVector criterionGradientX
    (const Plato::ScalarVector & aControl,
     const std::string         & aName)
    {
        auto tItr = mCriteria.find(aName);
        if (tItr == mCriteria.end())
        {
            THROWERR(std::string("Criterion with tag '") + aName + "' is not defined in critria map.");
        }

        Plato::Dual tDual;
        Plato::Primal tCurrentState, tPreviousState;
        auto tDirectory = std::string("solution_history");
        auto tSolutionHistoryPaths = Plato::omega_h::read_pvtu_file_paths(tDirectory);
        if(tSolutionHistoryPaths.size() != static_cast<size_t>(mNumForwardSolveTimeSteps + 1))
        {
            THROWERR(std::string("Number of time steps read from the '") + tDirectory
                 + "' directory does not match the expected value: '" + std::to_string(mNumForwardSolveTimeSteps + 1) + "'.")
        }

        Plato::ScalarVector tTotalDerivative("total derivative", mSpatialModel.Mesh.nverts());
        for(auto tItr = tSolutionHistoryPaths.rbegin(); tItr != tSolutionHistoryPaths.rend() - 1; tItr++)
        {
            // set fields for the current primal state
            auto tCurrentStateIndex = (tSolutionHistoryPaths.size() - 1u) - std::distance(tSolutionHistoryPaths.rbegin(), tItr);
            tCurrentState.scalar("time step index", tCurrentStateIndex);
            this->setPrimal(tSolutionHistoryPaths, tCurrentState);
            this->setCriticalTimeStep(tCurrentState);

            // set fields for the previous primal state
            auto tPreviousStateIndex = tCurrentStateIndex + 1u;
            tPreviousState.scalar("time step index", tPreviousStateIndex);
            if(tPreviousStateIndex != tSolutionHistoryPaths.size())
            {
                this->setPrimal(tSolutionHistoryPaths, tPreviousState);
                this->setCriticalTimeStep(tPreviousState);
            }

            // set adjoint state
            this->setDual(tDual);

            // update adjoint states
            if(mCalculateHeatTransfer)
            {
                this->updateTemperatureAdjoint(aName, aControl, tCurrentState, tPreviousState, tDual);
            }
            this->updateCorrectorAdjoint(aName, aControl, tCurrentState, tPreviousState, tDual);
            this->updatePressureAdjoint(aName, aControl, tCurrentState, tPreviousState, tDual);
            this->updatePredictorAdjoint(aControl, tCurrentState, tPreviousState, tDual);

            // update total derivative with respect to control variables
            this->updateTotalDerivativeWrtConfig(aName, aControl, tCurrentState, tDual, tTotalDerivative);

            this->saveDual(tDual);
        }
        return tTotalDerivative;
    }

    /******************************************************************************//**
     * \fn Plato::Scalar criterionGradientX
     *
     * \brief Evaluate criterion gradient with respect to configuration variables.
     * \param [in] aControl  vector of design/optimization variables
     * \param [in] aSolution Plato database with state solutions
     * \param [in] aName     criterion name/identifier
     * \return criterion gradient with respect to configuration variables
     *
     **********************************************************************************/
    Plato::ScalarVector criterionGradientX
    (const Plato::ScalarVector & aControl,
     const Plato::Solutions    & aSolution,
     const std::string         & aName)
    {
        return (this->criterionGradientX(aControl, aName));
    }

private:
    /******************************************************************************//**
     * \fn bool writeOutput
     *
     * \brief Return boolean used to determine if state solution will be written to
     *   visualization file.
     * \param [in] aIteration current solver iteration
     * \return boolean (true = output to file; false = skip output to file)
     *
     **********************************************************************************/
    bool writeOutput(const Plato::OrdinalType aIteration) const
    {
        auto tWrite = false;
        if(mOutputFrequency > static_cast<Plato::OrdinalType>(0))
        {
            auto tModulo = (aIteration + static_cast<Plato::OrdinalType>(1)) % mOutputFrequency;
            tWrite = tModulo == static_cast<Plato::OrdinalType>(0) ? true : false;
        }
        return tWrite;
    }

    /******************************************************************************//**
     * \fn void readCurrentFields
     *
     * \brief Read current states from visualization file.
     * \param [in]     aPath   visualization file path
     * \param [in/out] aStates primal state solution database
     *
     **********************************************************************************/
    void readCurrentFields
    (const Omega_h::filesystem::path & aPath,
           Plato::Primal             & aPrimal)
    {
        Plato::FieldTags tFieldTags;
        tFieldTags.set("Velocity", "current velocity");
        tFieldTags.set("Pressure", "current pressure");
        tFieldTags.set("Predictor", "current predictor");
        if(mCalculateHeatTransfer)
        {
            tFieldTags.set("Temperature", "current temperature");
        }
        Plato::omega_h::read_fields<Omega_h::VERT>(mSpatialModel.Mesh, aPath, tFieldTags, aPrimal);
    }

    /******************************************************************************//**
     * \fn void readPreviousFields
     *
     * \brief Read previous states from visualization file.
     * \param [in]     aPath   visualization file path
     * \param [in/out] aStates primal state solution database
     *
     **********************************************************************************/
    void readPreviousFields
    (const Omega_h::filesystem::path & aPath,
           Plato::Primal             & aPrimal)
    {
        Plato::FieldTags tFieldTags;
        tFieldTags.set("Velocity", "previous velocity");
        tFieldTags.set("Pressure", "previous pressure");
        if(mCalculateHeatTransfer)
        {
            tFieldTags.set("Temperature", "previous temperature");
        }
        Plato::omega_h::read_fields<Omega_h::VERT>(mSpatialModel.Mesh, aPath, tFieldTags, aPrimal);
    }

    /******************************************************************************//**
     * \fn void setPrimal
     *
     * \brief Set primal state solution database for the current optimization iteration.
     * \param [in]     aPath   list with paths to visualization files
     * \param [in/out] aPrimal primal state solution database
     *
     **********************************************************************************/
    void setPrimal
    (const std::vector<Omega_h::filesystem::path> & aPaths,
           Plato::Primal                          & aPrimal)
    {
        auto tTimeStepIndex = static_cast<size_t>(aPrimal.scalar("time step index"));
        this->readCurrentFields(aPaths[tTimeStepIndex], aPrimal);
        this->readPreviousFields(aPaths[tTimeStepIndex - 1u], aPrimal);
    }

    /******************************************************************************//**
     * \fn void setCriticalTimeStep
     *
     * \brief Set critical time step for the current optimization iteration.
     * \param [in/out] aPrimal primal state solution database
     *
     **********************************************************************************/
    void setCriticalTimeStep
    (Plato::Primal& aPrimal)
    {
        auto tTimeStepIndex = static_cast<size_t>(aPrimal.scalar("time step index"));
        Plato::ScalarVector tCriticalTimeStep("critical time step", 1);
        auto tHostCriticalTimeStep = Kokkos::create_mirror(tCriticalTimeStep);
        tHostCriticalTimeStep(0) = mCriticalTimeStepHistory[tTimeStepIndex];
        Kokkos::deep_copy(tCriticalTimeStep, tHostCriticalTimeStep);
        aPrimal.vector("critical time step", tCriticalTimeStep);
    }

    /******************************************************************************//**
     * \fn void setSolution
     *
     * \brief Set solution database.
     * \return solution database
     *
     **********************************************************************************/
    Plato::Solutions setSolution()
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

    /******************************************************************************//**
     * \fn void setInitialConditions
     *
     * \brief Set initial conditions for pressure, temperature and veloctity fields.
     * \param [in] aPrimal primal state database
     * \param [in] aWriter interface to allow output to a VTK visualization file
     *
     **********************************************************************************/
    void setInitialConditions
    (Plato::Primal & aPrimal,
     Omega_h::vtk::Writer& aWriter)
    {
        const Plato::Scalar tTime = 0.0;
        const Plato::OrdinalType tTimeStep = 0;
        mCriticalTimeStepHistory.push_back(0.0);
        aPrimal.scalar("time step index", tTimeStep);
        aPrimal.scalar("critical velocity lower bound", mCriticalVelocityLowerBound);

        Plato::ScalarVector tVelBcValues;
        Plato::LocalOrdinalVector tVelBcDofs;
        mVelocityEssentialBCs.get(tVelBcDofs, tVelBcValues, tTime);
        auto tPreviouVel = Kokkos::subview(mVelocity, tTimeStep, Kokkos::ALL());
        Plato::cbs::enforce_boundary_condition(tVelBcDofs, tVelBcValues, tPreviouVel);
        aPrimal.vector("previous velocity", tPreviouVel);

        Plato::ScalarVector tPressBcValues;
        Plato::LocalOrdinalVector tPressBcDofs;
        mPressureEssentialBCs.get(tPressBcDofs, tPressBcValues, tTime);
        auto tPreviousPress = Kokkos::subview(mPressure, tTimeStep, Kokkos::ALL());
        Plato::cbs::enforce_boundary_condition(tPressBcDofs, tPressBcValues, tPreviousPress);
        aPrimal.vector("previous pressure", tPreviousPress);

        auto tPreviousPred = Kokkos::subview(mPredictor, tTimeStep, Kokkos::ALL());
        aPrimal.vector("previous predictor", tPreviousPred);

        if(mCalculateHeatTransfer)
        {
            Plato::ScalarVector tTempBcValues;
            Plato::LocalOrdinalVector tTempBcDofs;
            mTemperatureEssentialBCs.get(tTempBcDofs, tTempBcValues, tTime);
            auto tPreviousTemp  = Kokkos::subview(mTemperature, tTimeStep, Kokkos::ALL());
            Plato::cbs::enforce_boundary_condition(tTempBcDofs, tTempBcValues, tPreviousTemp);
            aPrimal.vector("previous temperature", tPreviousTemp);

            aPrimal.scalar("thermal diffusivity", mCriticalThermalDiffusivity);
            aPrimal.scalar("kinematic viscocity", mCriticalKinematicViscocity);
        }

        if(this->writeOutput(tTimeStep))
        {
            this->write(aPrimal, aWriter);
        }
    }

    /******************************************************************************//**
     * \fn void printIteration
     *
     * \brief Print current iteration diagnostics to diagnostic file.
     * \param [in] aPrimal primal state database
     *
     **********************************************************************************/
    void printIteration
    (const Plato::Primal & aPrimal)
    {
        if(Plato::Comm::rank(mMachine) == 0)
        {
            if(mPrintDiagnostics)
            {
                std::stringstream tMsg;
                auto tCriticalTimeStep = aPrimal.vector("critical time step");
                auto tHostCriticalTimeStep = Kokkos::create_mirror(tCriticalTimeStep);
                Kokkos::deep_copy(tHostCriticalTimeStep, tCriticalTimeStep);
                const Plato::OrdinalType tTimeStepIndex = aPrimal.scalar("time step index");
                tMsg << "*************************************************************************************\n";
                tMsg << "* Critical Time Step: " << tHostCriticalTimeStep(0) << "\n";
                tMsg << "* CFD Quasi-Implicit Solver Iteration: " << tTimeStepIndex << "\n";
                tMsg << "*************************************************************************************\n";
                Plato::append_text_to_file(tMsg, mDiagnostics);
            }
        }
    }

    /******************************************************************************//**
     * \fn void areDianosticsEnabled
     *
     * \brief Check if diagnostics are enabled, if true, open diagnostic file.
     * \param [in] aInputs input file database
     *
     **********************************************************************************/
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

    /******************************************************************************//**
     * \fn void initialize
     *
     * \brief Initialize member data.
     * \param [in] aInputs input file database
     *
     **********************************************************************************/
    void initialize
    (Teuchos::ParameterList & aInputs)
    {
        this->allocatePrimalStates();
        this->areDianosticsEnabled(aInputs);
        this->parseNewtonSolverInputs(aInputs);
        this->parseConvergenceCriteria(aInputs);
        this->parseTimeIntegratorInputs(aInputs);
        this->setHeatTransferEquation(aInputs);
        this->allocateOptimizationMetadata(aInputs);
    }

    /******************************************************************************//**
     * \fn void setCriticalFluidProperties
     *
     * \brief Set fluid properties used to calculate the critical time step for heat
     *   transfer applications.
     * \param [in] aInputs input file database
     *
     **********************************************************************************/
    void setCriticalFluidProperties(Teuchos::ParameterList &aInputs)
    {
        mCriticalThermalDiffusivity = Plato::teuchos::parse_max_material_property<Plato::Scalar>
            (aInputs, "Thermal Properties", "Thermal Diffusivity", mSpatialModel.Domains);
	Plato::is_positive_finite_number(mCriticalThermalDiffusivity, "Thermal Diffusivity");
        mCriticalKinematicViscocity = Plato::teuchos::parse_max_material_property<Plato::Scalar>
            (aInputs, "Viscous Properties", "Kinematic Viscocity", mSpatialModel.Domains);
	Plato::is_positive_finite_number(mCriticalKinematicViscocity, "Kinematic Viscocity");
    }

    /******************************************************************************//**
     * \fn void setHeatTransferEquation
     *
     * \brief Set temperature equation vector function interface if heat transfer
     *   calculations are requested.
     * \param [in] aInputs input file database
     *
     **********************************************************************************/
    void setHeatTransferEquation
    (Teuchos::ParameterList & aInputs)
    {
        mCalculateHeatTransfer = Plato::Fluids::calculate_heat_transfer(aInputs);
        if(mCalculateHeatTransfer)
        {
            mTemperatureResidual = std::make_shared<Plato::Fluids::VectorFunction<typename PhysicsT::EnergyPhysicsT>>
                    ("Temperature", mSpatialModel, mDataMap, aInputs);
            this->setCriticalFluidProperties(aInputs);
        }
    }

    /******************************************************************************//**
     * \fn void parseNewtonSolverInputs
     *
     * \brief Parse Newton solver parameters from input file.
     * \param [in] aInputs input file database
     *
     **********************************************************************************/
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

    /******************************************************************************//**
     * \fn void parseTimeIntegratorInputs
     *
     * \brief Parse time integration scheme parameters from input file.
     * \param [in] aInputs input file database
     *
     **********************************************************************************/
    void parseTimeIntegratorInputs
    (Teuchos::ParameterList & aInputs)
    {
        if(aInputs.isSublist("Time Integration"))
        {
            auto tTimeIntegration = aInputs.sublist("Time Integration");
            mTimeStepDamping = tTimeIntegration.get<Plato::Scalar>("Damping", 1.0);
            mTimeStepSafetyFactor = tTimeIntegration.get<Plato::Scalar>("Safety Factor", 0.7);
        }
    }

    /******************************************************************************//**
     * \fn void parseConvergenceCriteria
     *
     * \brief Parse fluid solver's convergence criteria from input file.
     * \param [in] aInputs input file database
     *
     **********************************************************************************/
    void parseConvergenceCriteria
    (Teuchos::ParameterList & aInputs)
    {
        if(aInputs.isSublist("Convergence"))
        {
            auto tConvergence = aInputs.sublist("Convergence");
            mSteadyStateTolerance = tConvergence.get<Plato::Scalar>("Steady State Tolerance", 1e-5);
            mMaxSteadyStateIterations = tConvergence.get<Plato::OrdinalType>("Maximum Iterations", 1000);
            mOutputFrequency = tConvergence.get<Plato::OrdinalType>("Output Frequency", mMaxSteadyStateIterations + 1);
        }
    }

    /******************************************************************************//**
     * \fn void clear
     *
     * \brief Clear forward solver state data. This function is utilized only in
     *   optimization workflows since the solver is used in re-entrant mode.
     *
     **********************************************************************************/
    void clear()
    {
        mNumForwardSolveTimeSteps = 0;
        mCriticalTimeStepHistory.clear();
        Plato::blas2::fill(0.0, mPressure);
        Plato::blas2::fill(0.0, mVelocity);
        Plato::blas2::fill(0.0, mPredictor);
        Plato::blas2::fill(0.0, mTemperature);

        auto tDirectory = std::string("solution_history");
        Plato::filesystem::remove(tDirectory);
    }

    /******************************************************************************//**
     * \fn void checkProblemSetup
     *
     * \brief Check forward problem setup.
     *
     **********************************************************************************/
    void checkProblemSetup()
    {
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

    /******************************************************************************//**
     * \fn void allocateDualStates
     *
     * \brief Allocate dual state containers.
     *
     **********************************************************************************/
    void allocateDualStates()
    {
        constexpr auto tTimeSnapshotsStored = 2;
        auto tNumNodes = mSpatialModel.Mesh.nverts();
        mAdjointPressure = Plato::ScalarMultiVector("Adjoint Pressure Snapshots", tTimeSnapshotsStored, tNumNodes);
        mAdjointVelocity = Plato::ScalarMultiVector("Adjoint Velocity Snapshots", tTimeSnapshotsStored, tNumNodes * mNumVelDofsPerNode);
        mAdjointPredictor = Plato::ScalarMultiVector("Adjoint Predictor Snapshots", tTimeSnapshotsStored, tNumNodes * mNumVelDofsPerNode);

        if(mCalculateHeatTransfer)
        {
            mAdjointTemperature = Plato::ScalarMultiVector("Adjoint Temperature Snapshots", tTimeSnapshotsStored, tNumNodes);
        }
    }

    /******************************************************************************//**
     * \fn void allocatePrimalStates
     *
     * \brief Allocate primal state containers.
     *
     **********************************************************************************/
    void allocatePrimalStates()
    {
        constexpr auto tTimeSnapshotsStored = 2;
        auto tNumNodes = mSpatialModel.Mesh.nverts();
        mPressure    = Plato::ScalarMultiVector("Pressure Snapshots", tTimeSnapshotsStored, tNumNodes);
        mVelocity    = Plato::ScalarMultiVector("Velocity Snapshots", tTimeSnapshotsStored, tNumNodes * mNumVelDofsPerNode);
        mPredictor   = Plato::ScalarMultiVector("Predictor Snapshots", tTimeSnapshotsStored, tNumNodes * mNumVelDofsPerNode);
        mTemperature = Plato::ScalarMultiVector("Temperature Snapshots", tTimeSnapshotsStored, tNumNodes);
    }

    /******************************************************************************//**
     * \fn void allocateCriteriaList
     *
     * \brief Allocate criteria list.
     * \param [in] aInputs input file database
     *
     **********************************************************************************/
    void allocateCriteriaList(Teuchos::ParameterList &aInputs)
    {
        Plato::Fluids::CriterionFactory<PhysicsT> tScalarFuncFactory;

        auto tCriteriaParams = aInputs.sublist("Criteria");
        for(Teuchos::ParameterList::ConstIterator tIndex = tCriteriaParams.begin(); tIndex != tCriteriaParams.end(); ++tIndex)
        {
            const Teuchos::ParameterEntry& tEntry = tCriteriaParams.entry(tIndex);
            if(tEntry.isList() == false)
            {
                THROWERR("Parameter in Criteria block is not supported. Expect lists only.")
            }
            auto tName = tCriteriaParams.name(tIndex);
            auto tCriterion = tScalarFuncFactory.createCriterion(mSpatialModel, mDataMap, aInputs, tName);
            if( tCriterion != nullptr )
            {
                mCriteria[tName] = tCriterion;
            }
        }
    }

    /******************************************************************************//**
     * \fn void allocateOptimizationMetadata
     *
     * \brief Allocate optimization problem metadata.
     * \param [in] aInputs input file database
     *
     **********************************************************************************/
    void allocateOptimizationMetadata(Teuchos::ParameterList &aInputs)
    {
        if(aInputs.isSublist("Criteria"))
        {
            this->allocateDualStates();
            this->allocateCriteriaList(aInputs);
        }
    }

    /******************************************************************************//**
     * \fn void calculateVelocityMisfitNorm
     *
     * \brief Calculate velocity misfit euclidean norm.
     * \param [in] aPrimal primal state database
     *
     **********************************************************************************/
    Plato::Scalar calculateVelocityMisfitNorm
    (const Plato::Primal & aPrimal)
    {
        auto tNumNodes = mSpatialModel.Mesh.nverts();
        auto tCurrentVelocity = aPrimal.vector("current velocity");
        auto tPreviousVelocity = aPrimal.vector("previous velocity");
        auto tMisfitError = Plato::cbs::calculate_misfit_euclidean_norm<mNumVelDofsPerNode>(tNumNodes, tCurrentVelocity, tPreviousVelocity);
        auto tCurrentVelNorm = Plato::blas1::norm(tCurrentVelocity);
        auto tOutput = tMisfitError / tCurrentVelNorm;
        return tOutput;
    }

    /******************************************************************************//**
     * \fn void calculatePressureMisfitNorm
     *
     * \brief Calculate pressure misfit euclidean norm.
     * \param [in] aPrimal primal state database
     *
     **********************************************************************************/
    Plato::Scalar calculatePressureMisfitNorm
    (const Plato::Primal & aPrimal)
    {
        auto tNumNodes = mSpatialModel.Mesh.nverts();
        auto tCurrentPressure = aPrimal.vector("current pressure");
        auto tPreviousPressure = aPrimal.vector("previous pressure");
        auto tMisfitError = Plato::cbs::calculate_misfit_euclidean_norm<mNumPressDofsPerNode>(tNumNodes, tCurrentPressure, tPreviousPressure);
        auto tCurrentNorm = Plato::blas1::norm(tCurrentPressure);
        auto tOutput = tMisfitError / tCurrentNorm;
        return tOutput;
    }

    /******************************************************************************//**
     * \fn void printSteadyStateCriterion
     *
     * \brief Print steady state criterion to diagnostic file.
     * \param [in] aPrimal primal state database
     *
     **********************************************************************************/
    void printSteadyStateCriterion
    (const Plato::Primal & aPrimal)
    {
        if(Plato::Comm::rank(mMachine) == 0)
        {
            if(mPrintDiagnostics)
            {
                std::stringstream tMsg;
                auto tCriterion = aPrimal.scalar("current steady state criterion");
                tMsg << "\n-------------------------------------------------------------------------------------\n";
                tMsg << std::scientific << " Steady State Convergence: " << tCriterion << "\n";
                tMsg << "-------------------------------------------------------------------------------------\n\n";
                Plato::append_text_to_file(tMsg, mDiagnostics);
            }
        }
    }

    /******************************************************************************//**
     * \fn bool isFluidSolverDiverging
     *
     * \brief Check is fluid solver is diverging.
     * \param [in] aPrimal primal state database
     * \return boolean (true = diverging; false = not diverging)
     *
     **********************************************************************************/
    bool isFluidSolverDiverging
    (Plato::Primal & aPrimal)
    {
        auto tCurrentCriterion = aPrimal.scalar("current steady state criterion");
        if(!std::isfinite(tCurrentCriterion) || std::isnan(tCurrentCriterion))
        {
            return true;
        }
        return false;
    }

    /******************************************************************************//**
     * \fn bool checkStoppingCriteria
     *
     * \brief Check fluid solver stopping criterion.
     * \param [in] aPrimal primal state database
     * \return boolean (true = converged; false = did not coverge)
     *
     **********************************************************************************/
    bool checkStoppingCriteria
    (Plato::Primal & aPrimal)
    {
        bool tStop = false;
        const Plato::OrdinalType tTimeStepIndex = aPrimal.scalar("time step index");
        const auto tCriterionValue = this->calculatePressureMisfitNorm(aPrimal);
        aPrimal.scalar("current steady state criterion", tCriterionValue);
        this->printSteadyStateCriterion(aPrimal);


        if (tCriterionValue < mSteadyStateTolerance)
        {
            tStop = true;
        }
        else if (tTimeStepIndex >= mMaxSteadyStateIterations)
        {
            tStop = true;
        }
        else if(this->isFluidSolverDiverging(aPrimal))
        {
            tStop = true;
        }

        aPrimal.scalar("previous steady state criterion", tCriterionValue);

        return tStop;
    }

    /******************************************************************************//**
     * \fn void calculateCharacteristicElemSize
     *
     * \brief Calculate characteristic element size
     * \param [in] aPrimal primal state database
     *
     **********************************************************************************/
    void calculateCharacteristicElemSize
    (Plato::Primal & aPrimal)
    {
        auto tElemCharSizes =
            Plato::cbs::calculate_element_characteristic_sizes<mNumSpatialDims,mNumNodesPerCell>(mSpatialModel);
        aPrimal.vector("element characteristic size", tElemCharSizes);
    }

    /******************************************************************************//**
     * \fn Plato::Scalar calculateCriticalConvectiveTimeStep
     *
     * \brief Calculate critical convective time step.
     * \param [in] aPrimal   primal state database
     * \param [in] aVelocity velocity field
     * \return critical convective time step
     *
     **********************************************************************************/
    Plato::Scalar
    calculateCriticalConvectiveTimeStep
    (const Plato::Primal & aPrimal,
     const Plato::ScalarVector & aVelocity)
    {
        auto tElemCharSize = aPrimal.vector("element characteristic size");
        auto tVelMag = Plato::cbs::calculate_convective_velocity_magnitude<mNumNodesPerCell>(mSpatialModel, aVelocity);
        auto tCriticalTimeStep = Plato::cbs::calculate_critical_convective_time_step
            (mSpatialModel, tElemCharSize, tVelMag, mTimeStepSafetyFactor);
        return tCriticalTimeStep;
    }

    /******************************************************************************//**
     * \fn Plato::Scalar calculateCriticalDiffusionTimeStep
     *
     * \brief Calculate critical diffusive time step.
     * \param [in] aPrimal primal state database
     * \return critical diffusive time step
     *
     **********************************************************************************/
    Plato::Scalar
    calculateCriticalDiffusionTimeStep
    (const Plato::Primal & aPrimal)
    {
        auto tElemCharSize = aPrimal.vector("element characteristic size");
        auto tKinematicViscocity = aPrimal.scalar("kinematic viscocity");
        auto tThermalDiffusivity = aPrimal.scalar("thermal diffusivity");
        auto tCriticalTimeStep = Plato::cbs::calculate_critical_diffusion_time_step
            (tKinematicViscocity, tThermalDiffusivity, tElemCharSize, mTimeStepSafetyFactor);
        return tCriticalTimeStep;
    }

    /******************************************************************************//**
     * \fn Plato::Scalar calculateCriticalTimeStepUpperBound
     *
     * \brief Calculate critical time step upper bound.
     * \param [in] aPrimal primal state database
     * \return critical time step upper bound
     *
     **********************************************************************************/
    inline Plato::Scalar
    calculateCriticalTimeStepUpperBound
    (const Plato::Primal &aPrimal)
    {
        auto tElemCharSize = aPrimal.vector("element characteristic size");
        auto tVelLowerBound = aPrimal.scalar("critical velocity lower bound");
        auto tOutput = Plato::cbs::calculate_critical_time_step_upper_bound(tVelLowerBound, tElemCharSize);
        return tOutput;
    }

    /******************************************************************************//**
     * \fn Plato::ScalarVector criticalTimeStep
     *
     * \brief Calculate critical time step.
     * \param [in] aPrimal primal state database
     * \param [in] aVelocity velocity field
     * \return critical time step
     *
     **********************************************************************************/
    Plato::ScalarVector
    criticalTimeStep
    (const Plato::Primal & aPrimal,
     const Plato::ScalarVector & aVelocity)
    {
        Plato::ScalarVector tCriticalTimeStep("critical time step", 1);
        auto tHostCriticalTimeStep = Kokkos::create_mirror(tCriticalTimeStep);

        tHostCriticalTimeStep(0) = this->calculateCriticalConvectiveTimeStep(aPrimal, aVelocity);
        if(mCalculateHeatTransfer)
        {
            auto tCriticalDiffusionTimeStep = this->calculateCriticalDiffusionTimeStep(aPrimal);
            auto tMinCriticalTimeStep = std::min(tCriticalDiffusionTimeStep, tHostCriticalTimeStep(0));
            tHostCriticalTimeStep(0) = tMinCriticalTimeStep;
        }

        auto tCriticalTimeStepUpperBound = this->calculateCriticalTimeStepUpperBound(aPrimal);
        auto tMinCriticalTimeStep = std::min(tCriticalTimeStepUpperBound, tHostCriticalTimeStep(0));
        tHostCriticalTimeStep(0) = mTimeStepDamping * tMinCriticalTimeStep;
        mCriticalTimeStepHistory.push_back(tHostCriticalTimeStep(0));
        Kokkos::deep_copy(tCriticalTimeStep, tHostCriticalTimeStep);

        return tCriticalTimeStep;
    }

    /******************************************************************************//**
     * \fn Plato::ScalarVector initialCriticalTimeStep
     *
     * \brief Calculate initial critical time step.
     * \param [in] aPrimal primal state database
     * \return critical time step
     *
     **********************************************************************************/
    Plato::ScalarVector
    initialCriticalTimeStep
    (const Plato::Primal & aPrimal)
    {
        Plato::ScalarVector tBcValues;
        Plato::LocalOrdinalVector tBcDofs;
        mVelocityEssentialBCs.get(tBcDofs, tBcValues);
        auto tPreviousVelocity = aPrimal.vector("previous velocity");
        Plato::ScalarVector tInitialVelocity("initial velocity", tPreviousVelocity.size());
        Plato::blas1::update(1.0, tPreviousVelocity, 0.0, tInitialVelocity);
        Plato::cbs::enforce_boundary_condition(tBcDofs, tBcValues, tInitialVelocity);
        auto tCriticalTimeStep = this->criticalTimeStep(aPrimal, tInitialVelocity);
        return tCriticalTimeStep;
    }

    /******************************************************************************//**
     * \fn void checkCriticalTimeStep
     *
     * \brief Check critical time step, an runtime error is thrown if an unstable time step is detected.
     * \param [in] aPrimal primal state database
     *
     **********************************************************************************/
    void checkCriticalTimeStep
    (const Plato::Primal &aPrimal)
    {
        auto tCriticalTimeStep = aPrimal.vector("critical time step");
        auto tHostCriticalTimeStep = Kokkos::create_mirror(tCriticalTimeStep);
        Kokkos::deep_copy(tHostCriticalTimeStep, tCriticalTimeStep);
        if(tHostCriticalTimeStep(0) < std::numeric_limits<Plato::Scalar>::epsilon())
        {
            std::ostringstream tOutSStream;
            tOutSStream << tHostCriticalTimeStep(0);
            THROWERR(std::string("Unstable critical time step (dt = '") + tOutSStream.str()
                 + "') detected. Refine the finite element mesh or coarsen the steady state stopping tolerance.")
        }
    }

    /******************************************************************************//**
     * \fn void calculateCriticalTimeStep
     *
     * \brief Calculate critical time step.
     * \param [in\out] aPrimal primal state database
     *
     **********************************************************************************/
    void calculateCriticalTimeStep
    (Plato::Primal & aPrimal)
    {
        auto tIteration = aPrimal.scalar("time step index");
        if(tIteration > 1)
        {
            auto tPreviousVelocity = aPrimal.vector("previous velocity");
            auto tCriticalTimeStep = this->criticalTimeStep(aPrimal, tPreviousVelocity);
            aPrimal.vector("critical time step", tCriticalTimeStep);
        }
        else
        {
            auto tCriticalTimeStep = this->initialCriticalTimeStep(aPrimal);
            aPrimal.vector("critical time step", tCriticalTimeStep);
        }
    }

    /******************************************************************************//**
     * \fn void setDual
     *
     * \brief Set dual state database
     * \param [in\out] aDual dual state database
     *
     **********************************************************************************/
    void setDual
    (Plato::Dual& aDual)
    {
        constexpr auto tCurrentSnapshot = 1u;
        auto tCurrentAdjointVel = Kokkos::subview(mAdjointVelocity, tCurrentSnapshot, Kokkos::ALL());
        auto tCurrentAdjointPred = Kokkos::subview(mAdjointPredictor, tCurrentSnapshot, Kokkos::ALL());
        auto tCurrentAdjointPress = Kokkos::subview(mAdjointPressure, tCurrentSnapshot, Kokkos::ALL());
        aDual.vector("current velocity adjoint", tCurrentAdjointVel);
        aDual.vector("current pressure adjoint", tCurrentAdjointPress);
        aDual.vector("current predictor adjoint", tCurrentAdjointPred);

        constexpr auto tPreviousSnapshot = tCurrentSnapshot - 1u;
        auto tPreviouAdjointVel = Kokkos::subview(mAdjointVelocity, tPreviousSnapshot, Kokkos::ALL());
        auto tPreviousAdjointPred = Kokkos::subview(mAdjointPredictor, tPreviousSnapshot, Kokkos::ALL());
        auto tPreviousAdjointPress = Kokkos::subview(mAdjointPressure, tPreviousSnapshot, Kokkos::ALL());
        aDual.vector("previous velocity adjoint", tPreviouAdjointVel);
        aDual.vector("previous predictor adjoint", tPreviousAdjointPred);
        aDual.vector("previous pressure adjoint", tPreviousAdjointPress);

	    if(mCalculateHeatTransfer)
	    {
                auto tCurrentAdjointTemp = Kokkos::subview(mAdjointTemperature, tCurrentSnapshot, Kokkos::ALL());
                auto tPreviousAdjointTemp = Kokkos::subview(mAdjointTemperature, tPreviousSnapshot, Kokkos::ALL());
                aDual.vector("current temperature adjoint", tCurrentAdjointTemp);
                aDual.vector("previous temperature adjoint", tPreviousAdjointTemp);
	    }
    }

    /******************************************************************************//**
     * \fn void saveDual
     *
     * \brief Set previous dual state for the next iteration.
     * \param [in\out] aDual dual state database
     *
     **********************************************************************************/
    void saveDual
    (Plato::Dual & aDual)
    {
        constexpr auto tPreviousSnapshot = 0u;

        auto tCurrentAdjointVelocity = aDual.vector("current velocity adjoint");
        auto tPreviousAdjointVelocity = Kokkos::subview(mAdjointVelocity, tPreviousSnapshot, Kokkos::ALL());
        Plato::blas1::copy(tCurrentAdjointVelocity, tPreviousAdjointVelocity);

        auto tCurrentAdjointPressure = aDual.vector("current pressure adjoint");
        auto tPreviousAdjointPressure = Kokkos::subview(mAdjointPressure, tPreviousSnapshot, Kokkos::ALL());
        Plato::blas1::copy(tCurrentAdjointPressure, tPreviousAdjointPressure);

        auto tCurrentAdjointPredictor = aDual.vector("current predictor adjoint");
        auto tPreviousAdjointPredictor = Kokkos::subview(mAdjointPredictor, tPreviousSnapshot, Kokkos::ALL());
        Plato::blas1::copy(tCurrentAdjointPredictor, tPreviousAdjointPredictor);

        if(mCalculateHeatTransfer)
        {
            auto tCurrentAdjointTemperature = aDual.vector("current temperature adjoint");
            auto tPreviousAdjointTemperature = Kokkos::subview(mAdjointTemperature, tPreviousSnapshot, Kokkos::ALL());
            Plato::blas1::copy(tCurrentAdjointTemperature, tPreviousAdjointTemperature);
        }
    }

    /******************************************************************************//**
     * \fn void savePrimal
     *
     * \brief Set previous primal state for the next iteration.
     * \param [in\out] aPrimal primal state database
     *
     **********************************************************************************/
    void savePrimal
    (Plato::Primal & aPrimal)
    {
        constexpr auto tPreviousSnapshot = 0u;

        auto tCurrentVelocity = aPrimal.vector("current velocity");
        auto tPreviousVelocity = Kokkos::subview(mVelocity, tPreviousSnapshot, Kokkos::ALL());
        Plato::blas1::copy(tCurrentVelocity, tPreviousVelocity);

        auto tCurrentPressure = aPrimal.vector("current pressure");
        auto tPreviousPressure = Kokkos::subview(mPressure, tPreviousSnapshot, Kokkos::ALL());
        Plato::blas1::copy(tCurrentPressure, tPreviousPressure);

        auto tCurrentPredictor = aPrimal.vector("current predictor");
        auto tPreviousPredictor = Kokkos::subview(mPredictor, tPreviousSnapshot, Kokkos::ALL());
        Plato::blas1::copy(tCurrentPredictor, tPreviousPredictor);

        if(mCalculateHeatTransfer)
        {
            auto tCurrentTemperature = aPrimal.vector("current temperature");
            auto tPreviousTemperature = Kokkos::subview(mTemperature, tPreviousSnapshot, Kokkos::ALL());
            Plato::blas1::copy(tCurrentTemperature, tPreviousTemperature);
        }
    }

    /******************************************************************************//**
     * \fn void setPrimal
     *
     * \brief Set previous and current primal states.
     * \param [in\out] aPrimal primal state database
     *
     **********************************************************************************/
    void setPrimal
    (Plato::Primal & aPrimal)
    {
        constexpr Plato::OrdinalType tCurrentState = 1;
        auto tCurrentVel   = Kokkos::subview(mVelocity, tCurrentState, Kokkos::ALL());
        auto tCurrentPred  = Kokkos::subview(mPredictor, tCurrentState, Kokkos::ALL());
        auto tCurrentPress = Kokkos::subview(mPressure, tCurrentState, Kokkos::ALL());
        aPrimal.vector("current velocity", tCurrentVel);
        aPrimal.vector("current pressure", tCurrentPress);
        aPrimal.vector("current predictor", tCurrentPred);

        constexpr auto tPrevState = tCurrentState - 1;
        auto tPreviouVel = Kokkos::subview(mVelocity, tPrevState, Kokkos::ALL());
        auto tPreviousPred = Kokkos::subview(mPredictor, tPrevState, Kokkos::ALL());
        auto tPreviousPress = Kokkos::subview(mPressure, tPrevState, Kokkos::ALL());
        aPrimal.vector("previous velocity", tPreviouVel);
        aPrimal.vector("previous predictor", tPreviousPred);
        aPrimal.vector("previous pressure", tPreviousPress);

        auto tCurrentTemp = Kokkos::subview(mTemperature, tCurrentState, Kokkos::ALL());
        aPrimal.vector("current temperature", tCurrentTemp);
        auto tPreviousTemp = Kokkos::subview(mTemperature, tPrevState, Kokkos::ALL());
        aPrimal.vector("previous temperature", tPreviousTemp);
    }

    /******************************************************************************//**
     * \fn void printCorrectorSolverHeader
     *
     * \brief Print diagnostic header for velocity corrector solver.
     *
     **********************************************************************************/
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

    /******************************************************************************//**
     * \fn void updateCorrector
     *
     * \brief Solve for current velocity field using the Newton method.
     * \param [in]     aControl control/optimization variables
     * \param [in\out] aPrimal  primal state database
     *
     **********************************************************************************/
    void updateCorrector
    (const Plato::ScalarVector & aControl,
           Plato::Primal       & aPrimal)
    {
        this->printCorrectorSolverHeader();
        this->printNewtonHeader();

        auto tCurrentVelocity = aPrimal.vector("current velocity");
        Plato::blas1::fill(0.0, tCurrentVelocity);

        // calculate current residual and jacobian matrix
        auto tJacobian = mCorrectorResidual.gradientCurrentVel(aControl, aPrimal);

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
        Plato::Scalar tInitialNormStep = 0.0, tInitialNormResidual = 0.0;
        Plato::ScalarVector tDeltaCorrector("delta corrector", tCurrentVelocity.size());
        while(true)
        {
            aPrimal.scalar("newton iteration", tIteration);

            auto tResidual = mCorrectorResidual.value(aControl, aPrimal);
            Plato::blas1::scale(-1.0, tResidual);
            Plato::blas1::fill(0.0, tDeltaCorrector);
            tSolver->solve(*tJacobian, tDeltaCorrector, tResidual);
            Plato::blas1::update(1.0, tDeltaCorrector, 1.0, tCurrentVelocity);

            auto tNormResidual = Plato::blas1::norm(tResidual);
            auto tNormStep = Plato::blas1::norm(tDeltaCorrector);
            if(tIteration <= 1)
            {
                tInitialNormStep = tNormStep;
                tInitialNormResidual = tNormResidual;
            }
            tNormStep = tNormStep / tInitialNormStep;
            aPrimal.scalar("norm step", tNormStep);
            tNormResidual = tNormResidual / tInitialNormResidual;
            aPrimal.scalar("norm residual", tNormResidual);

            this->printNewtonDiagnostics(aPrimal);
            if(tNormStep <= mCorrectorTolerance || tIteration >= mMaxCorrectorIterations)
            {
                break;
            }

            tIteration++;
        }
        Plato::cbs::enforce_boundary_condition(tBcDofs, tBcValues, tCurrentVelocity);
    }

    /******************************************************************************//**
     * \fn void printNewtonHeader
     *
     * \brief Print Newton solver header to diagnostics text file.
     *
     **********************************************************************************/
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

    /******************************************************************************//**
     * \fn void printPredictorSolverHeader
     *
     * \brief Print velocity predictor solver header to diagnostics text file.
     *
     **********************************************************************************/
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

    /******************************************************************************//**
     * \fn void printNewtonDiagnostics
     *
     * \brief Print Newton's solver diagnostics to text file.
     * \param [in] aPrimal  primal state database
     *
     **********************************************************************************/
    void printNewtonDiagnostics
    (Plato::Primal & aPrimal)
    {
        if(Plato::Comm::rank(mMachine) == 0)
        {
            if(mPrintDiagnostics)
            {
                std::stringstream tMsg;
                auto tNormStep = aPrimal.scalar("norm step");
                auto tNormResidual = aPrimal.scalar("norm residual");
                Plato::OrdinalType tIteration = aPrimal.scalar("newton iteration");
                tMsg << tIteration << std::setw(24) << std::scientific << tNormStep << std::setw(18) << tNormResidual << "\n";
                Plato::append_text_to_file(tMsg, mDiagnostics);
            }
        }
    }

    /******************************************************************************//**
     * \fn void updatePredictor
     *
     * \brief Solve for current velocity predictor field using the Newton method.
     * \param [in]     aControl control/optimization variables
     * \param [in\out] aPrimal  primal state database
     *
     **********************************************************************************/
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
        Plato::Scalar tInitialNormStep = 0.0, tInitialNormResidual = 0.0;
        Plato::ScalarVector tDeltaPredictor("delta predictor", tCurrentPredictor.size());
        while(true)
        {
            aStates.scalar("newton iteration", tIteration);

            Plato::blas1::fill(0.0, tDeltaPredictor);
            Plato::blas1::scale(-1.0, tResidual);
            tSolver->solve(*tJacobian, tDeltaPredictor, tResidual);
            Plato::blas1::update(1.0, tDeltaPredictor, 1.0, tCurrentPredictor);

            auto tNormResidual = Plato::blas1::norm(tResidual);
            auto tNormStep = Plato::blas1::norm(tDeltaPredictor);
            if(tIteration <= 1)
            {
                tInitialNormStep = tNormStep;
                tInitialNormResidual = tNormResidual;
            }
            tNormStep = tNormStep / tInitialNormStep;
            aStates.scalar("norm step", tNormStep);
            tNormResidual = tNormResidual / tInitialNormResidual;
            aStates.scalar("norm residual", tNormResidual);

            this->printNewtonDiagnostics(aStates);
            if(tNormStep <= mPredictorTolerance || tIteration >= mMaxPredictorIterations)
            {
                break;
            }

            tResidual = mPredictorResidual.value(aControl, aStates);

            tIteration++;
        }
    }

    /******************************************************************************//**
     * \fn void printPressureSolverHeader
     *
     * \brief Print pressure solver header to diagnostics text file.
     *
     **********************************************************************************/
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

    /******************************************************************************//**
     * \fn void updatePressure
     *
     * \brief Solve for current pressure field using the Newton method.
     * \param [in]     aControl control/optimization variables
     * \param [in\out] aPrimal  primal state database
     *
     **********************************************************************************/
    void updatePressure
    (const Plato::ScalarVector & aControl,
           Plato::Primal       & aStates)
    {
        this->printPressureSolverHeader();
        this->printNewtonHeader();

        auto tCurrentPressure = aStates.vector("current pressure");
        Plato::blas1::fill(0.0, tCurrentPressure);

        // prepare constraints dofs
        Plato::ScalarVector tBcValues;
        Plato::LocalOrdinalVector tBcDofs;
        mPressureEssentialBCs.get(tBcDofs, tBcValues);

        // create linear solver
        auto tParamList = mInputs.sublist("Linear Solver");
        Plato::SolverFactory tSolverFactory(tParamList);
        auto tSolver = tSolverFactory.create(mSpatialModel.Mesh, mMachine, mNumPressDofsPerNode);

        Plato::OrdinalType tIteration = 1;
        Plato::Scalar tInitialNormStep = 0.0, tInitialNormResidual = 0.0;
        Plato::ScalarVector tDeltaPressure("delta pressure", tCurrentPressure.size());
        while(true)
        {
            aStates.scalar("newton iteration", tIteration);

            auto tResidual = mPressureResidual.value(aControl, aStates);
            Plato::blas1::scale(-1.0, tResidual);
            auto tJacobian = mPressureResidual.gradientCurrentPress(aControl, aStates);

            Plato::Scalar tScale = (tIteration == 1) ? 1.0 : 0.0;
            Plato::apply_constraints<mNumPressDofsPerNode>(tBcDofs, tBcValues, tJacobian, tResidual, tScale);
            Plato::blas1::fill(0.0, tDeltaPressure);
            tSolver->solve(*tJacobian, tDeltaPressure, tResidual);
            Plato::blas1::update(1.0, tDeltaPressure, 1.0, tCurrentPressure);

            auto tNormResidual = Plato::blas1::norm(tResidual);
            auto tNormStep = Plato::blas1::norm(tDeltaPressure);
            if(tIteration <= 1)
            {
                tInitialNormStep = tNormStep;
                tInitialNormResidual = tNormResidual;
            }
            tNormStep = tNormStep / tInitialNormStep;
            aStates.scalar("norm step", tNormStep);
            tNormResidual = tNormResidual / tInitialNormResidual;
            aStates.scalar("norm residual", tNormResidual);

            this->printNewtonDiagnostics(aStates);
            if(tNormStep <= mPressureTolerance || tIteration >= mMaxPressureIterations)
            //if(tNormResidual <= mPressureTolerance || tIteration >= mMaxPressureIterations)
            {
                break;
            }

            tIteration++;
        }
    }

    /******************************************************************************//**
     * \fn void printTemperatureSolverHeader
     *
     * \brief Print temperature solver header to diagnostics text file.
     *
     **********************************************************************************/
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

    /******************************************************************************//**
     * \fn void updateTemperature
     *
     * \brief Solve for current temperature field using the Newton method.
     * \param [in]     aControl control/optimization variables
     * \param [in\out] aPrimal  primal state database
     *
     **********************************************************************************/
    void updateTemperature
    (const Plato::ScalarVector & aControl,
           Plato::Primal       & aStates)
    {
        this->printTemperatureSolverHeader();
        this->printNewtonHeader();

        auto tCurrentTemperature = aStates.vector("current temperature");
        Plato::blas1::fill(0.0, tCurrentTemperature);

        // apply constraints
        Plato::ScalarVector tBcValues;
        Plato::LocalOrdinalVector tBcDofs;
        mTemperatureEssentialBCs.get(tBcDofs, tBcValues);

        // solve energy equation (consistent or mass lumped)
        auto tParamList = mInputs.sublist("Linear Solver");
        Plato::SolverFactory tSolverFactory(tParamList);
        auto tSolver = tSolverFactory.create(mSpatialModel.Mesh, mMachine, mNumTempDofsPerNode);

        Plato::OrdinalType tIteration = 1;
        Plato::Scalar tInitialNormStep = 0.0, tInitialNormResidual = 0.0;
        Plato::ScalarVector tDeltaTemperature("delta temperature", tCurrentTemperature.size());
        while(true)
        {
            aStates.scalar("newton iteration", tIteration);

            // update residual and jacobian
            auto tResidual = mTemperatureResidual->value(aControl, aStates);
            Plato::blas1::scale(-1.0, tResidual);
            auto tJacobian = mTemperatureResidual->gradientCurrentTemp(aControl, aStates);

            // solve system of equations
            Plato::Scalar tScale = (tIteration == 1) ? 1.0 : 0.0;
            Plato::apply_constraints<mNumTempDofsPerNode>(tBcDofs, tBcValues, tJacobian, tResidual, tScale);
            Plato::blas1::fill(0.0, tDeltaTemperature);
            tSolver->solve(*tJacobian, tDeltaTemperature, tResidual);
            Plato::blas1::update(1.0, tDeltaTemperature, 1.0, tCurrentTemperature);

            // calculate stopping criteria
            auto tNormResidual = Plato::blas1::norm(tResidual);
            auto tNormStep = Plato::blas1::norm(tDeltaTemperature);
            if(tIteration <= 1)
            {
                tInitialNormStep = tNormStep;
                tInitialNormResidual = tNormResidual;
            }
            tNormStep = tNormStep / tInitialNormStep;
            aStates.scalar("norm step", tNormStep);
            tNormResidual = tNormResidual / tInitialNormResidual;
            aStates.scalar("norm residual", tNormResidual);

            // check stopping criteria
            this->printNewtonDiagnostics(aStates);
            if(tNormResidual < mTemperatureTolerance || tIteration >= mMaxTemperatureIterations)
            {
                break;
            }

            tIteration++;
        }
    }

    /******************************************************************************//**
     * \fn void updatePredictorAdjoint
     *
     * \brief Solve for the current velocity predictor adjoint field using the Newton method.
     * \param [in]     aControl        control/optimization variables
     * \param [in]     aCurrentPrimal  current primal state database
     * \param [in]     aPreviousPrimal previous primal state database
     * \param [in/out] aDual           current dual state database
     *
     **********************************************************************************/
    void updatePredictorAdjoint
    (const Plato::ScalarVector & aControl,
     const Plato::Primal       & aCurrentPrimal,
     const Plato::Primal       & aPreviousPrimal,
           Plato::Dual         & aDual)
    {
        auto tCurrentPredictorAdjoint = aDual.vector("current predictor adjoint");
        Plato::blas1::fill(0.0, tCurrentPredictorAdjoint);

        // add PDE contribution from current state to right hand side adjoint vector
        auto tCurrentVelocityAdjoint = aDual.vector("current velocity adjoint");
        auto tJacCorrectorResWrtPredictor = mCorrectorResidual.gradientPredictor(aControl, aCurrentPrimal);
        Plato::ScalarVector tRHS("right hand side", tCurrentVelocityAdjoint.size());
        Plato::MatrixTimesVectorPlusVector(tJacCorrectorResWrtPredictor, tCurrentVelocityAdjoint, tRHS);

        auto tCurrentPressureAdjoint = aDual.vector("current pressure adjoint");
        auto tGradResPressWrtPredictor = mPressureResidual.gradientPredictor(aControl, aCurrentPrimal);
        Plato::MatrixTimesVectorPlusVector(tGradResPressWrtPredictor, tCurrentPressureAdjoint, tRHS);
        Plato::blas1::scale(-1.0, tRHS);

        // solve adjoint system of equations
        auto tParamList = mInputs.sublist("Linear Solver");
        Plato::SolverFactory tSolverFactory(tParamList);
        auto tSolver = tSolverFactory.create(mSpatialModel.Mesh, mMachine, mNumVelDofsPerNode);
        auto tJacobianPredictor = mPredictorResidual.gradientPredictor(aControl, aCurrentPrimal);
        tSolver->solve(*tJacobianPredictor, tCurrentPredictorAdjoint, tRHS);
    }

    /******************************************************************************//**
     * \fn void updatePressureAdjoint
     *
     * \brief Solve for the current pressure adjoint field using the Newton method.
     * \param [in]     aControl        control/optimization variables
     * \param [in]     aCurrentPrimal  current primal state database
     * \param [in]     aPreviousPrimal previous primal state database
     * \param [in/out] aDual           current dual state database
     *
     **********************************************************************************/
    void updatePressureAdjoint
    (const std::string         & aName,
     const Plato::ScalarVector & aControl,
     const Plato::Primal       & aCurrentPrimal,
     const Plato::Primal       & aPreviousPrimal,
           Plato::Dual         & aDual)
    {
        // initialize data
        auto tCurrentTimeStepIndex = static_cast<Plato::OrdinalType>(aCurrentPrimal.scalar("time step index"));
        auto tCurrentPressAdjoint = aDual.vector("current pressure adjoint");
        Plato::blas1::fill(0.0, tCurrentPressAdjoint);

        // add objective function contribution to right hand side adjoint vector
        auto tNumDofs = mSpatialModel.Mesh.nverts();
        Plato::ScalarVector tRightHandSide("right hand side vector", tNumDofs);
        if(tCurrentTimeStepIndex == mNumForwardSolveTimeSteps)
        {
            auto tPartialObjWrtCurrentPressure = mCriteria[aName]->gradientCurrentPress(aControl, aCurrentPrimal);
            Plato::blas1::update(1.0, tPartialObjWrtCurrentPressure, 0.0, tRightHandSide);
        }

        // add PDE contribution from current state to right hand side adjoint vector
        auto tCurrentVelocityAdjoint = aDual.vector("current velocity adjoint");
        auto tJacCorrectorResWrtCurPress = mCorrectorResidual.gradientCurrentPress(aControl, aCurrentPrimal);
        Plato::MatrixTimesVectorPlusVector(tJacCorrectorResWrtCurPress, tCurrentVelocityAdjoint, tRightHandSide);

        // add PDE contribution from previous state to right hand side adjoint vector
        if(tCurrentTimeStepIndex != mNumForwardSolveTimeSteps)
        {
            auto tPreviousPressureAdjoint = aDual.vector("previous pressure adjoint");
            auto tJacPressResWrtPrevPress = mPressureResidual.gradientPreviousPress(aControl, aPreviousPrimal);
            Plato::MatrixTimesVectorPlusVector(tJacPressResWrtPrevPress, tPreviousPressureAdjoint, tRightHandSide);

            auto tPreviousVelocityAdjoint = aDual.vector("previous velocity adjoint");
            auto tJacCorrectorResWrtPrevVel = mCorrectorResidual.gradientPreviousPress(aControl, aPreviousPrimal);
            Plato::MatrixTimesVectorPlusVector(tJacCorrectorResWrtPrevVel, tPreviousVelocityAdjoint, tRightHandSide);
        }
        Plato::blas1::scale(-1.0, tRightHandSide);

        // prepare constraints dofs
        Plato::ScalarVector tBcValues;
        Plato::LocalOrdinalVector tBcDofs;
        mPressureEssentialBCs.get(tBcDofs, tBcValues);
        Plato::blas1::fill(0.0, tBcValues);

        // solve adjoint system of equations
        auto tParamList = mInputs.sublist("Linear Solver");
        Plato::SolverFactory tSolverFactory(tParamList);
        auto tSolver = tSolverFactory.create(mSpatialModel.Mesh, mMachine, mNumPressDofsPerNode);
        auto tJacPressResWrtCurPress = mPressureResidual.gradientCurrentPress(aControl, aCurrentPrimal);
        Plato::apply_constraints<mNumPressDofsPerNode>(tBcDofs, tBcValues, tJacPressResWrtCurPress, tRightHandSide);
        tSolver->solve(*tJacPressResWrtCurPress, tCurrentPressAdjoint, tRightHandSide);
    }

    /******************************************************************************//**
     * \fn void updateTemperatureAdjoint
     *
     * \brief Solve for the current temperature adjoint field using the Newton method.
     * \param [in]     aControl        control/optimization variables
     * \param [in]     aCurrentPrimal  current primal state database
     * \param [in]     aPreviousPrimal previous primal state database
     * \param [in/out] aDual           current dual state database
     *
     **********************************************************************************/
    void updateTemperatureAdjoint
    (const std::string         & aName,
     const Plato::ScalarVector & aControl,
     const Plato::Primal       & aCurrentPrimal,
     const Plato::Primal       & aPreviousPrimal,
           Plato::Dual         & aDual)
    {
        // initialize data
        auto tCurrentTimeStepIndex = static_cast<Plato::OrdinalType>(aCurrentPrimal.scalar("time step index"));
        auto tCurrentTempAdjoint = aDual.vector("current temperature adjoint");
        Plato::blas1::fill(0.0, tCurrentTempAdjoint);

        // add objective function contribution to right hand side adjoint vector
        auto tNumDofs = mSpatialModel.Mesh.nverts();
        Plato::ScalarVector tRightHandSide("right hand side vector", tNumDofs);
        if(tCurrentTimeStepIndex == mNumForwardSolveTimeSteps)
        {
            auto tPartialObjWrtCurrentTemperature = mCriteria[aName]->gradientCurrentTemp(aControl, aCurrentPrimal);
            Plato::blas1::update(1.0, tPartialObjWrtCurrentTemperature, 0.0, tRightHandSide);
        }

        // add PDE contribution from previous state to right hand side adjoint vector
        if(tCurrentTimeStepIndex != mNumForwardSolveTimeSteps)
        {
            auto tPreviousPredAdjoint = aDual.vector("previous predictor adjoint");
            auto tGradResPredWrtPreviousTemp = mPredictorResidual.gradientPreviousTemp(aControl, aPreviousPrimal);
            Plato::MatrixTimesVectorPlusVector(tGradResPredWrtPreviousTemp, tPreviousPredAdjoint, tRightHandSide);

            if(mCalculateHeatTransfer)
            {
                auto tPreviousTempAdjoint = aDual.vector("previous temperature adjoint");
                auto tJacTempResWrtPreviousTemp = mTemperatureResidual->gradientPreviousTemp(aControl, aPreviousPrimal);
                Plato::MatrixTimesVectorPlusVector(tJacTempResWrtPreviousTemp, tPreviousTempAdjoint, tRightHandSide);
            }
        }
        Plato::blas1::scale(-1.0, tRightHandSide);

        // prepare constraints dofs
        Plato::ScalarVector tBcValues;
        Plato::LocalOrdinalVector tBcDofs;
        mTemperatureEssentialBCs.get(tBcDofs, tBcValues);
        Plato::blas1::fill(0.0, tBcValues);

        // solve adjoint system of equations
        auto tParamList = mInputs.sublist("Linear Solver");
        Plato::SolverFactory tSolverFactory(tParamList);
        auto tSolver = tSolverFactory.create(mSpatialModel.Mesh, mMachine, mNumTempDofsPerNode);
        auto tJacobianCurrentTemp = mTemperatureResidual->gradientCurrentTemp(aControl, aCurrentPrimal);
        Plato::apply_constraints<mNumTempDofsPerNode>(tBcDofs, tBcValues, tJacobianCurrentTemp, tRightHandSide);
        tSolver->solve(*tJacobianCurrentTemp, tCurrentTempAdjoint, tRightHandSide);
    }

    /******************************************************************************//**
     * \fn void updateCorrectorAdjoint
     *
     * \brief Solve for the current velocity adjoint field using the Newton method.
     * \param [in]     aControl        control/optimization variables
     * \param [in]     aCurrentPrimal  current primal state database
     * \param [in]     aPreviousPrimal previous primal state database
     * \param [in/out] aDual           current dual state database
     *
     **********************************************************************************/
    void updateCorrectorAdjoint
    (const std::string         & aName,
     const Plato::ScalarVector & aControl,
     const Plato::Primal       & aCurrentPrimalState,
     const Plato::Primal       & aPreviousPrimalState,
           Plato::Dual         & aDual)
    {
        // initialize data
        auto tCurrentTimeStepIndex = static_cast<Plato::OrdinalType>(aCurrentPrimalState.scalar("time step index"));
        auto tCurrentVelocityAdjoint = aDual.vector("current velocity adjoint");
        Plato::blas1::fill(0.0, tCurrentVelocityAdjoint);

        // add objective function contribution to right hand side adjoint vector
        auto tNumDofs = mSpatialModel.Mesh.nverts() * mNumVelDofsPerNode;
        Plato::ScalarVector tRightHandSide("right hand side vector", tNumDofs);
        if(tCurrentTimeStepIndex == mNumForwardSolveTimeSteps)
        {
            auto tPartialObjFuncWrtCurrentVel = mCriteria[aName]->gradientCurrentVel(aControl, aCurrentPrimalState);
            Plato::blas1::update(1.0, tPartialObjFuncWrtCurrentVel, 0.0, tRightHandSide);
        }

        // add PDE contribution from current state to right hand side adjoint vector
        if(mCalculateHeatTransfer)
        {
            auto tCurrentTempAdjoint = aDual.vector("current temperature adjoint");
            auto tJacTempResWrtCurVel = mTemperatureResidual->gradientCurrentVel(aControl, aCurrentPrimalState);
            Plato::MatrixTimesVectorPlusVector(tJacTempResWrtCurVel, tCurrentTempAdjoint, tRightHandSide);
        }
	

        // add PDE contribution from previous state to right hand side adjoint vector
        if(tCurrentTimeStepIndex != mNumForwardSolveTimeSteps)
        {
            auto tPreviousPredictorAdjoint = aDual.vector("previous predictor adjoint");
            auto tJacPredResWrtPrevVel = mPredictorResidual.gradientPreviousVel(aControl, aPreviousPrimalState);
            Plato::MatrixTimesVectorPlusVector(tJacPredResWrtPrevVel, tPreviousPredictorAdjoint, tRightHandSide);

            auto tPreviousPressureAdjoint = aDual.vector("previous pressure adjoint");
            auto tJacPressResWrtPrevVel = mPressureResidual.gradientPreviousVel(aControl, aPreviousPrimalState);
            Plato::MatrixTimesVectorPlusVector(tJacPressResWrtPrevVel, tPreviousPressureAdjoint, tRightHandSide);

            auto tPreviousVelocityAdjoint = aDual.vector("previous velocity adjoint");
            auto tJacCorrectorResWrtPrevVel = mCorrectorResidual.gradientPreviousVel(aControl, aPreviousPrimalState);
            Plato::MatrixTimesVectorPlusVector(tJacCorrectorResWrtPrevVel, tPreviousVelocityAdjoint, tRightHandSide);
        }
        Plato::blas1::scale(-1.0, tRightHandSide);

        // prepare constraints dofs
        Plato::ScalarVector tBcValues;
        Plato::LocalOrdinalVector tBcDofs;
        mVelocityEssentialBCs.get(tBcDofs, tBcValues);
        Plato::blas1::fill(0.0, tBcValues);

        // solve adjoint system of equations
        auto tParamList = mInputs.sublist("Linear Solver");
        Plato::SolverFactory tSolverFactory(tParamList);
        auto tSolver = tSolverFactory.create(mSpatialModel.Mesh, mMachine, mNumVelDofsPerNode);
        auto tJacCorrectorResWrtCurVel = mCorrectorResidual.gradientCurrentVel(aControl, aCurrentPrimalState);
        Plato::set_dofs_values(tBcDofs, tRightHandSide, 0.0);
        tSolver->solve(*tJacCorrectorResWrtCurVel, tCurrentVelocityAdjoint, tRightHandSide);
    }

    /******************************************************************************//**
     * \fn void updateTotalDerivativeWrtControl
     *
     * \brief Update total derivative of the criterion with respect to control variables.
     * \param [in]     aName            criterion name
     * \param [in]     aControl         control/optimization variables
     * \param [in]     aCurrentPrimal   current primal state database
     * \param [in]     aDual            current dual state database
     * \param [in/out] aTotalDerivative total derivative
     *
     **********************************************************************************/
    void  updateTotalDerivativeWrtControl
    (const std::string         & aName,
     const Plato::ScalarVector & aControl,
     const Plato::Primal       & aCurrentPrimal,
     const Plato::Dual         & aDual,
           Plato::ScalarVector & aTotalDerivative)
    {
        auto tCurrentTimeStepIndex = static_cast<Plato::OrdinalType>(aCurrentPrimal.scalar("time step index"));
        if(tCurrentTimeStepIndex == mNumForwardSolveTimeSteps)
        {
            auto tGradCriterionWrtControl = mCriteria[aName]->gradientControl(aControl, aCurrentPrimal);
            Plato::blas1::update(1.0, tGradCriterionWrtControl, 1.0, aTotalDerivative);
        }

        auto tCurrentPredictorAdjoint = aDual.vector("current predictor adjoint");
        auto tGradResPredWrtControl = mPredictorResidual.gradientControl(aControl, aCurrentPrimal);
        Plato::MatrixTimesVectorPlusVector(tGradResPredWrtControl, tCurrentPredictorAdjoint, aTotalDerivative);

        auto tCurrentPressureAdjoint = aDual.vector("current pressure adjoint");
        auto tGradResPressWrtControl = mPressureResidual.gradientControl(aControl, aCurrentPrimal);
        Plato::MatrixTimesVectorPlusVector(tGradResPressWrtControl, tCurrentPressureAdjoint, aTotalDerivative);

        auto tCurrentVelocityAdjoint = aDual.vector("current velocity adjoint");
        auto tGradResVelWrtControl = mCorrectorResidual.gradientControl(aControl, aCurrentPrimal);
        Plato::MatrixTimesVectorPlusVector(tGradResVelWrtControl, tCurrentVelocityAdjoint, aTotalDerivative);

        if(mCalculateHeatTransfer)
        {
            auto tCurrentTemperatureAdjoint = aDual.vector("current temperature adjoint");
            auto tGradResTempWrtControl = mTemperatureResidual->gradientControl(aControl, aCurrentPrimal);
            Plato::MatrixTimesVectorPlusVector(tGradResTempWrtControl, tCurrentTemperatureAdjoint, aTotalDerivative);
        }
    }

    /******************************************************************************//**
     * \fn void updateTotalDerivativeWrtConfig
     *
     * \brief Update total derivative of the criterion with respect to the configuration variables.
     * \param [in]     aName            criterion name
     * \param [in]     aControl         control/optimization variables
     * \param [in]     aCurrentPrimal   current primal state database
     * \param [in]     aDual            current dual state database
     * \param [in/out] aTotalDerivative total derivative
     *
     **********************************************************************************/
    void updateTotalDerivativeWrtConfig
    (const std::string         & aName,
     const Plato::ScalarVector & aControl,
     const Plato::Primal       & aCurrentPrimal,
     const Plato::Dual         & aDual,
           Plato::ScalarVector & aTotalDerivative)
    {
        auto tCurrentTimeStepIndex = static_cast<Plato::OrdinalType>(aCurrentPrimal.scalar("time step index"));
        if(tCurrentTimeStepIndex == mNumForwardSolveTimeSteps)
        {
            auto tGradCriterionWrtConfig = mCriteria[aName]->gradientConfig(aControl, aCurrentPrimal);
            Plato::blas1::update(1.0, tGradCriterionWrtConfig, 1.0, aTotalDerivative);
        }

        auto tCurrentPredictorAdjoint = aDual.vector("current predictor adjoint");
        auto tGradResPredWrtConfig = mPredictorResidual.gradientConfig(aControl, aCurrentPrimal);
        Plato::MatrixTimesVectorPlusVector(tGradResPredWrtConfig, tCurrentPredictorAdjoint, aTotalDerivative);

        auto tCurrentPressureAdjoint = aDual.vector("current pressure adjoint");
        auto tGradResPressWrtConfig = mPressureResidual.gradientConfig(aControl, aCurrentPrimal);
        Plato::MatrixTimesVectorPlusVector(tGradResPressWrtConfig, tCurrentPressureAdjoint, aTotalDerivative);

        auto tCurrentVelocityAdjoint = aDual.vector("current velocity adjoint");
        auto tGradResVelWrtConfig = mCorrectorResidual.gradientConfig(aControl, aCurrentPrimal);
        Plato::MatrixTimesVectorPlusVector(tGradResVelWrtConfig, tCurrentVelocityAdjoint, aTotalDerivative);

        if(mCalculateHeatTransfer)
        {
            auto tCurrentTemperatureAdjoint = aDual.vector("current temperature adjoint");
            auto tGradResTempWrtConfig = mTemperatureResidual->gradientConfig(aControl, aCurrentPrimal);
            Plato::MatrixTimesVectorPlusVector(tGradResTempWrtConfig, tCurrentTemperatureAdjoint, aTotalDerivative);
        }
    }
};
// class QuasiImplicit

}
// namespace Hyperbolic

}
//namespace Plato

namespace ComputationalFluidDynamicsTests
{

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, setState)
{
    // set xml file inputs
    Teuchos::RCP<Teuchos::ParameterList> tInputs =
        Teuchos::getParametersFromXmlString(
            "<ParameterList name='Plato Problem'>"
            "  <ParameterList name='Hyperbolic'>"
            "    <Parameter name='Heat Transfer' type='string' value='None'/>"
            "    <ParameterList  name='Momentum Conservation'>"
            "      <Parameter  name='Stabilization Constant' type='double' value='1.0'/>"
            "    </ParameterList>"
            "    <ParameterList  name='Dimensionless Properties'>"
            "      <Parameter  name='Reynolds Number'  type='double'  value='100'/>"
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
            "      <Parameter  name='Value'    type='double' value='1'/>"
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
            "    <Parameter name='Safety Factor' type='double' value='0.7'/>"
            "  </ParameterList>"
            "  <ParameterList  name='Linear Solver'>"
            "    <Parameter name='Solver Stack' type='string' value='Epetra'/>"
            "  </ParameterList>"
            "  <ParameterList  name='Convergence'>"
            "    <Parameter name='Output Frequency' type='int' value='1'/>"
            "    <Parameter name='Maximum Iterations' type='int' value='2'/>"
            "  </ParameterList>"
            "</ParameterList>"
            );

    // build mesh, spatial domain, and spatial model
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,10,10);
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

    // read pvtu file path
    Plato::Primal tPrimal;
    auto tPaths = Plato::omega_h::read_pvtu_file_paths("solution_history");
    TEST_EQUALITY(3u, tPaths.size());

    // fill field tags and names
    Plato::FieldTags tCurrentFieldTags;
    tCurrentFieldTags.set("Velocity", "current velocity");
    tCurrentFieldTags.set("Pressure", "current pressure");
    Plato::FieldTags tPreviousFieldTags;
    tPreviousFieldTags.set("Velocity", "previous velocity");
    tPreviousFieldTags.set("Pressure", "previous pressure");

    // read fields and set state struct
    auto tTol = 1e-2;
    std::vector<Plato::Scalar> tGoldMaxVel = {1.0, 1.0, 1.0};
    std::vector<Plato::Scalar> tGoldMinVel = {0.0, -0.0795658, -0.0916226};
    std::vector<Plato::Scalar> tGoldMaxPress = {0.0, 6.40268, 4.27897};
    std::vector<Plato::Scalar> tGoldMinPress = {0.0, 0.0, 0.0};
    for(auto tItr = tPaths.rbegin(); tItr != tPaths.rend() - 1; tItr++)
    {
	auto tCurrentIndex = (tPaths.size() - 1u) - std::distance(tPaths.rbegin(), tItr);
	Plato::omega_h::read_fields<Omega_h::VERT>(*tMesh, tPaths[tCurrentIndex], tCurrentFieldTags, tPrimal);

        Plato::Scalar tMaxVel = 0;
        Plato::blas1::max(tPrimal.vector("current velocity"), tMaxVel);
        TEST_FLOATING_EQUALITY(tGoldMaxVel[tCurrentIndex], tMaxVel, tTol);
        Plato::Scalar tMinVel = 0;
        Plato::blas1::min(tPrimal.vector("current velocity"), tMinVel);
        TEST_FLOATING_EQUALITY(tGoldMinVel[tCurrentIndex], tMinVel, tTol);

        Plato::Scalar tMaxPress = 0;
        Plato::blas1::max(tPrimal.vector("current pressure"), tMaxPress);
        TEST_FLOATING_EQUALITY(tGoldMaxPress[tCurrentIndex], tMaxPress, tTol);
        Plato::Scalar tMinPress = 0;
        Plato::blas1::min(tPrimal.vector("current pressure"), tMinPress);
        TEST_FLOATING_EQUALITY(tGoldMinPress[tCurrentIndex], tMinPress, tTol);

	auto tPreviousIndex = tCurrentIndex - 1u;
	Plato::omega_h::read_fields<Omega_h::VERT>(*tMesh, tPaths[tPreviousIndex], tPreviousFieldTags, tPrimal);

        tMaxVel = 0;
        Plato::blas1::max(tPrimal.vector("previous velocity"), tMaxVel);
        TEST_FLOATING_EQUALITY(tGoldMaxVel[tPreviousIndex], tMaxVel, tTol);
        tMinVel = 0;
        Plato::blas1::min(tPrimal.vector("previous velocity"), tMinVel);
        TEST_FLOATING_EQUALITY(tGoldMinVel[tPreviousIndex], tMinVel, tTol);

        tMaxPress = 0;
        Plato::blas1::max(tPrimal.vector("previous pressure"), tMaxPress);
        TEST_FLOATING_EQUALITY(tGoldMaxPress[tPreviousIndex], tMaxPress, tTol);
        tMinPress = 0;
        Plato::blas1::min(tPrimal.vector("previous pressure"), tMinPress);
        TEST_FLOATING_EQUALITY(tGoldMinPress[tPreviousIndex], tMinPress, tTol);
    }

    std::system("rm -rf solution_history");
    std::system("rm -f cfd_solver_diagnostics.txt");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ReadFields)
{
    // set xml file inputs
    Teuchos::RCP<Teuchos::ParameterList> tInputs =
        Teuchos::getParametersFromXmlString(
            "<ParameterList name='Plato Problem'>"
            "  <ParameterList name='Hyperbolic'>"
            "    <Parameter name='Heat Transfer' type='string' value='None'/>"
            "    <ParameterList  name='Momentum Conservation'>"
            "      <Parameter  name='Stabilization Constant' type='double' value='1.0'/>"
            "    </ParameterList>"
            "    <ParameterList  name='Dimensionless Properties'>"
            "      <Parameter  name='Reynolds Number'  type='double'  value='100'/>"
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
            "      <Parameter  name='Value'    type='double' value='1'/>"
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
            "    <Parameter name='Safety Factor' type='double' value='0.7'/>"
            "  </ParameterList>"
            "  <ParameterList  name='Linear Solver'>"
            "    <Parameter name='Solver Stack' type='string' value='Epetra'/>"
            "  </ParameterList>"
            "  <ParameterList  name='Convergence'>"
            "    <Parameter name='Output Frequency' type='int' value='1'/>"
            "    <Parameter name='Maximum Iterations' type='int' value='2'/>"
            "  </ParameterList>"
            "</ParameterList>"
            );

    // build mesh, spatial domain, and spatial model
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,10,10);
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

    // read pvtu file path
    Plato::Primal tCurrentState;
    auto tPaths = Plato::omega_h::read_pvtu_file_paths("solution_history");
    TEST_EQUALITY(3u, tPaths.size());

    // fill field tags and names
    Plato::FieldTags tFieldTags;
    tFieldTags.set("Velocity", "current velocity");
    tFieldTags.set("Pressure", "current pressure");
    tFieldTags.set("Predictor", "current predictor");

    auto tTol = 1e-2;
    std::vector<Plato::Scalar> tGoldMaxVel = {1.0, 1.0, 1.0};
    std::vector<Plato::Scalar> tGoldMinVel = {0.0, -0.0795658, -0.0916226};
    std::vector<Plato::Scalar> tGoldMaxPred = {0.0, 0.921744, 1.01069};
    std::vector<Plato::Scalar> tGoldMinPred = {0.0, -0.125735, -0.113359};
    std::vector<Plato::Scalar> tGoldMaxPress = {0.0, 6.40268, 4.27897};
    std::vector<Plato::Scalar> tGoldMinPress = {0.0, 0.0, 0.0};
    for(auto tItr = tPaths.rbegin(); tItr != tPaths.rend(); tItr++)
    {
	auto tIndex = (tPaths.size() - 1u) - std::distance(tPaths.rbegin(), tItr);
	Plato::omega_h::read_fields<Omega_h::VERT>(*tMesh, tPaths[tIndex], tFieldTags, tCurrentState);

        Plato::Scalar tMaxVel = 0;
        Plato::blas1::max(tCurrentState.vector("current velocity"), tMaxVel);
        TEST_FLOATING_EQUALITY(tGoldMaxVel[tIndex], tMaxVel, tTol);
        Plato::Scalar tMinVel = 0;
        Plato::blas1::min(tCurrentState.vector("current velocity"), tMinVel);
        TEST_FLOATING_EQUALITY(tGoldMinVel[tIndex], tMinVel, tTol);

        Plato::Scalar tMaxPred = 0;
        Plato::blas1::max(tCurrentState.vector("current predictor"), tMaxPred);
        TEST_FLOATING_EQUALITY(tGoldMaxPred[tIndex], tMaxPred, tTol);
        Plato::Scalar tMinPred = 0;
        Plato::blas1::min(tCurrentState.vector("current predictor"), tMinPred);
        TEST_FLOATING_EQUALITY(tGoldMinPred[tIndex], tMinPred, tTol);

        Plato::Scalar tMaxPress = 0;
        Plato::blas1::max(tCurrentState.vector("current pressure"), tMaxPress);
        TEST_FLOATING_EQUALITY(tGoldMaxPress[tIndex], tMaxPress, tTol);
        Plato::Scalar tMinPress = 0;
        Plato::blas1::min(tCurrentState.vector("current pressure"), tMinPress);
        TEST_FLOATING_EQUALITY(tGoldMinPress[tIndex], tMinPress, tTol);
    }

    std::system("rm -rf solution_history");
    std::system("rm -f cfd_solver_diagnostics.txt");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, Test_Omega_h_ReadParallel)
{
    // set xml file inputs
    Teuchos::RCP<Teuchos::ParameterList> tInputs =
        Teuchos::getParametersFromXmlString(
            "<ParameterList name='Plato Problem'>"
            "  <ParameterList name='Hyperbolic'>"
            "    <Parameter name='Heat Transfer' type='string' value='None'/>"
            "    <ParameterList  name='Momentum Conservation'>"
            "      <Parameter  name='Stabilization Constant' type='double' value='1.0'/>"
            "    </ParameterList>"
            "    <ParameterList  name='Dimensionless Properties'>"
            "      <Parameter  name='Reynolds Number'  type='double'  value='100'/>"
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
            "      <Parameter  name='Value'    type='double' value='1'/>"
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
            "    <Parameter name='Safety Factor' type='double' value='0.7'/>"
            "  </ParameterList>"
            "  <ParameterList  name='Linear Solver'>"
            "    <Parameter name='Solver Stack' type='string' value='Epetra'/>"
            "  </ParameterList>"
            "  <ParameterList  name='Convergence'>"
            "    <Parameter name='Output Frequency' type='int' value='1'/>"
            "    <Parameter name='Maximum Iterations' type='int' value='2'/>"
            "  </ParameterList>"
            "</ParameterList>"
            );

    // build mesh, spatial domain, and spatial model
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,10,10);
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

    // read pvtu file paths
    auto tPaths = Plato::omega_h::read_pvtu_file_paths("solution_history");
    TEST_EQUALITY(3u, tPaths.size());

    auto tTol = 1e-2;
    std::vector<Plato::Scalar> tGoldMaxVel = {1.0, 1.0, 1.0};
    std::vector<Plato::Scalar> tGoldMinVel = {0.0, -0.0795658, -0.0916226};
    std::vector<Plato::Scalar> tGoldMaxPred = {0.0, 0.921744, 1.01069};
    std::vector<Plato::Scalar> tGoldMinPred = {0.0, -0.125735, -0.113359};
    std::vector<Plato::Scalar> tGoldMaxPress = {0.0, 6.40268, 4.27897};
    std::vector<Plato::Scalar> tGoldMinPress = {0.0, 0.0, 0.0};
    for(auto& tPath : tPaths)
    {
	auto tIndex = &tPath - &tPaths[0];
	Omega_h::Mesh tReadMesh(tMesh->library());
	Omega_h::vtk::read_parallel(tPath, tMesh->library()->world(), &tReadMesh);

	auto tVelocity = Plato::omega_h::read_metadata_from_mesh(tReadMesh, Omega_h::VERT, "Velocity");
	TEST_EQUALITY(242, tVelocity.size());
        Plato::Scalar tMaxVel = 0;
        Plato::blas1::max(tVelocity, tMaxVel);
        TEST_FLOATING_EQUALITY(tGoldMaxVel[tIndex], tMaxVel, tTol);
        Plato::Scalar tMinVel = 0;
        Plato::blas1::min(tVelocity, tMinVel);
        TEST_FLOATING_EQUALITY(tGoldMinVel[tIndex], tMinVel, tTol);

	auto tPredictor = Plato::omega_h::read_metadata_from_mesh(tReadMesh, Omega_h::VERT, "Predictor");
	TEST_EQUALITY(242, tPredictor.size());
        Plato::Scalar tMaxPred = 0;
        Plato::blas1::max(tPredictor, tMaxPred);
        TEST_FLOATING_EQUALITY(tGoldMaxPred[tIndex], tMaxPred, tTol);
        Plato::Scalar tMinPred = 0;
        Plato::blas1::min(tPredictor, tMinPred);
        TEST_FLOATING_EQUALITY(tGoldMinPred[tIndex], tMinPred, tTol);

	auto tPressure = Plato::omega_h::read_metadata_from_mesh(tReadMesh, Omega_h::VERT, "Pressure");
	TEST_EQUALITY(121, tPressure.size());
        Plato::Scalar tMaxPress = 0;
        Plato::blas1::max(tPressure, tMaxPress);
        TEST_FLOATING_EQUALITY(tGoldMaxPress[tIndex], tMaxPress, tTol);
        Plato::Scalar tMinPress = 0;
        Plato::blas1::min(tPressure, tMinPress);
        TEST_FLOATING_EQUALITY(tGoldMinPress[tIndex], tMinPress, tTol);
    }

    std::system("rm -rf solution_history");
    std::system("rm -f cfd_solver_diagnostics.txt");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ReadPvtuFilePaths)
{
    // set xml file inputs
    Teuchos::RCP<Teuchos::ParameterList> tInputs =
        Teuchos::getParametersFromXmlString(
            "<ParameterList name='Plato Problem'>"
            "  <ParameterList name='Hyperbolic'>"
            "    <Parameter name='Heat Transfer' type='string' value='None'/>"
            "    <ParameterList  name='Momentum Conservation'>"
            "      <Parameter  name='Stabilization Constant' type='double' value='1.0'/>"
            "    </ParameterList>"
            "    <ParameterList  name='Dimensionless Properties'>"
            "      <Parameter  name='Reynolds Number'  type='double'  value='100'/>"
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
            "      <Parameter  name='Value'    type='double' value='1'/>"
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
            "    <Parameter name='Safety Factor' type='double' value='0.7'/>"
            "  </ParameterList>"
            "  <ParameterList  name='Linear Solver'>"
            "    <Parameter name='Solver Stack' type='string' value='Epetra'/>"
            "  </ParameterList>"
            "  <ParameterList  name='Convergence'>"
            "    <Parameter name='Output Frequency' type='int' value='1'/>"
            "    <Parameter name='Maximum Iterations' type='int' value='10'/>"
            "  </ParameterList>"
            "</ParameterList>"
            );

    // build mesh, spatial domain, and spatial model
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,10,10);
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

    // read pvtu file paths
    auto tPaths = Plato::omega_h::read_pvtu_file_paths("solution_history");
    TEST_EQUALITY(11u, tPaths.size());

    // test paths
    std::vector<std::string> tGold = 
        {"solution_history/steps/step_0/pieces.pvtu", "solution_history/steps/step_1/pieces.pvtu", "solution_history/steps/step_2/pieces.pvtu",
	 "solution_history/steps/step_3/pieces.pvtu", "solution_history/steps/step_4/pieces.pvtu", "solution_history/steps/step_5/pieces.pvtu",
	 "solution_history/steps/step_6/pieces.pvtu", "solution_history/steps/step_7/pieces.pvtu", "solution_history/steps/step_8/pieces.pvtu",
	 "solution_history/steps/step_9/pieces.pvtu", "solution_history/steps/step_10/pieces.pvtu"};
    for(auto& tPath : tPaths)
    {
	auto tIndex = &tPath - &tPaths[0];
	TEST_EQUALITY(tGold[tIndex], tPath.string());
    }

    std::system("rm -rf solution_history");
    std::system("rm -f cfd_solver_diagnostics.txt");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, IsothermalFlowOnChannel_Re100_CriterionGradient)
{
    // set xml file inputs
    Teuchos::RCP<Teuchos::ParameterList> tInputs =
        Teuchos::getParametersFromXmlString(
            "<ParameterList name='Plato Problem'>"
            "  <ParameterList name='Criteria'>"
            "    <ParameterList name='Inlet Average Surface Pressure'>"
            "      <Parameter name='Type' type='string' value='Scalar Function'/> "
            "      <Parameter  name='Sides' type='Array(string)' value='{x-}'/>"
            "      <Parameter name='Scalar Function Type' type='string' value='Average Surface Pressure'/>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList name='Hyperbolic'>"
            "    <Parameter name='Scenario' type='string' value='Density TO'/>"
            "    <Parameter name='Heat Transfer' type='string' value='None'/>"
            "    <ParameterList  name='Momentum Conservation'>"
            "      <Parameter  name='Stabilization Constant' type='double' value='1'/>"
            "    </ParameterList>"
            "    <ParameterList  name='Dimensionless Properties'>"
            "      <Parameter  name='Reynolds Number'  type='double'  value='100'/>"
            "      <Parameter  name='Impermeability Number'  type='double'  value='1'/>"
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
            "      <Parameter  name='Value'    type='double' value='1'/>"
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
            "    <Parameter name='Safety Factor' type='double' value='0.7'/>"
            "  </ParameterList>"
            "  <ParameterList  name='Linear Solver'>"
            "    <Parameter name='Solver Stack' type='string' value='Epetra'/>"
            "  </ParameterList>"
            "  <ParameterList  name='Convergence'>"
            "    <Parameter name='Output Frequency' type='int' value='1'/>"
            "    <Parameter name='Maximum Iterations' type='int' value='5'/>"
            "    <Parameter name='Steady State Tolerance' type='double' value='1e-3'/>"
            "  </ParameterList>"
            "</ParameterList>"
            );

    // build mesh, spatial domain, and spatial model
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,10,10);
    auto tMeshSets = PlatoUtestHelpers::get_box_mesh_sets(tMesh.operator*());
    Plato::SpatialDomain tDomain(tMesh.operator*(), tMeshSets, "box");
    tDomain.cellOrdinals("body");

    // create communicator
    MPI_Comm tMyComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &tMyComm);
    Plato::Comm::Machine tMachine(tMyComm);

    // create and test gradient wrt control for incompressible cfd problem
    constexpr auto tSpaceDim = 2;
    Plato::Fluids::QuasiImplicit<Plato::IncompressibleFluids<tSpaceDim>> tProblem(*tMesh, tMeshSets, *tInputs, tMachine);
    auto tError = Plato::test_criterion_grad_wrt_control(tProblem, *tMesh, "Inlet Average Surface Pressure", 4, 6);
    TEST_ASSERT(tError < 1e-4);

    std::system("rm -rf solution_history");
    std::system("rm -f cfd_solver_diagnostics.txt");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, NaturalConvectionSquareEnclosure_Ra1e3_CheckCriterionGradient)
{
    // set xml file inputs
    Teuchos::RCP<Teuchos::ParameterList> tInputs =
        Teuchos::getParametersFromXmlString(
            "<ParameterList name='Plato Problem'>"
            "  <ParameterList name='Criteria'>"
            "    <ParameterList name='Average Surface Temperature'>"
            "      <Parameter name='Type' type='string' value='Scalar Function'/> "
            "      <Parameter  name='Sides' type='Array(string)' value='{y+}'/>"
            "      <Parameter name='Scalar Function Type' type='string' value='Average Surface Temperature'/>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList name='Hyperbolic'>"
            "    <Parameter name='Scenario' type='string' value='Density TO'/>"
            "    <Parameter name='Heat Transfer' type='string' value='Natural'/>"
            "    <ParameterList  name='Momentum Conservation'>"
            "      <Parameter  name='Stabilization Constant' type='double' value='1'/>"
            "    </ParameterList>"
            "    <ParameterList  name='Dimensionless Properties'>"
            "      <Parameter  name='Prandtl Number'  type='double' value='0.7'/>"
            "      <Parameter  name='Impermeability Number'  type='double'  value='1'/>"
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
            "  <ParameterList name='Material Models'>"
            "    <ParameterList name='Air'>"
            "      <ParameterList name='Thermal Properties'>"
            "        <Parameter  name='Thermal Diffusivity' type='double' value='2.1117e-5'/>"
            "        <Parameter  name='Thermal Diffusivity Ratio' type='double' value='0.5'/>"
            "      </ParameterList>"
            "      <ParameterList name='Viscous Properties'>"
            "        <Parameter  name='Kinematic Viscocity' type='double' value='1.5111e-5'/>"
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
            "    <Parameter name='Temperature Tolerance' type='double' value='1e-4'/>"
            "  </ParameterList>"
            "  <ParameterList  name='Time Integration'>"
            "    <Parameter name='Safety Factor' type='double' value='0.1'/>"
            "    <Parameter name='Safety Factor' type='double' value='0.4'/>"
            "  </ParameterList>"
            "  <ParameterList  name='Linear Solver'>"
            "    <Parameter name='Solver Stack' type='string' value='Epetra'/>"
            "  </ParameterList>"
                "  <ParameterList  name='Convergence'>"
                "    <Parameter name='Output Frequency' type='int' value='1'/>"
                "    <Parameter name='Maximum Iterations' type='int' value='5'/>"
                "    <Parameter name='Steady State Tolerance' type='double' value='1e-5'/>"
                "  </ParameterList>"
            "</ParameterList>"
            );

    // build mesh, spatial domain, and spatial model
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,40, 40);
    auto tMeshSets = PlatoUtestHelpers::get_box_mesh_sets(tMesh.operator*());
    Plato::SpatialDomain tDomain(tMesh.operator*(), tMeshSets, "box");
    tDomain.cellOrdinals("body");

    // create communicator
    MPI_Comm tMyComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &tMyComm);
    Plato::Comm::Machine tMachine(tMyComm);

    // create and test gradient wrt control for incompressible cfd problem
    constexpr auto tSpaceDim = 2;
    Plato::Fluids::QuasiImplicit<Plato::IncompressibleFluids<tSpaceDim>> tProblem(*tMesh, tMeshSets, *tInputs, tMachine);
    auto tError = Plato::test_criterion_grad_wrt_control(tProblem, *tMesh, "Average Surface Temperature", 3, 5);
    TEST_ASSERT(tError < 1e-4);

    std::system("rm -rf solution_history");
    std::system("rm -f cfd_solver_diagnostics.txt");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, IsothermalFlowOnChannel_Re100_CriterionValue)
{
    // set xml file inputs
    Teuchos::RCP<Teuchos::ParameterList> tInputs =
        Teuchos::getParametersFromXmlString(
            "<ParameterList name='Plato Problem'>"
            "  <ParameterList name='Criteria'>"
            "    <ParameterList name='Outlet Average Surface Pressure'>"
            "      <Parameter name='Type' type='string' value='Scalar Function'/> "
            "      <Parameter  name='Sides' type='Array(string)' value='{x+}'/>"
            "      <Parameter name='Scalar Function Type' type='string' value='Average Surface Pressure'/>"
            "    </ParameterList>"
            "    <ParameterList name='Inlet Average Surface Pressure'>"
            "      <Parameter name='Type' type='string' value='Scalar Function'/> "
            "      <Parameter  name='Sides' type='Array(string)' value='{x-}'/>"
            "      <Parameter name='Scalar Function Type' type='string' value='Average Surface Pressure'/>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList name='Hyperbolic'>"
            "    <Parameter name='Heat Transfer' type='string' value='None'/>"
            "    <ParameterList  name='Momentum Conservation'>"
            "      <Parameter  name='Stabilization Constant' type='double' value='1.0'/>"
            "    </ParameterList>"
            "    <ParameterList  name='Dimensionless Properties'>"
            "      <Parameter  name='Reynolds Number'  type='double'  value='100'/>"
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
            "      <Parameter  name='Value'    type='double' value='1'/>"
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
            "    <Parameter name='Safety Factor' type='double' value='0.7'/>"
            "  </ParameterList>"
            "  <ParameterList  name='Linear Solver'>"
            "    <Parameter name='Solver Stack' type='string' value='Epetra'/>"
            "  </ParameterList>"
            "  <ParameterList  name='Convergence'>"
            "    <Parameter name='Output Frequency' type='int' value='1'/>"
            "    <Parameter name='Steady State Tolerance' type='double' value='1e-5'/>"
            "  </ParameterList>"
            "</ParameterList>"
            );

    // build mesh, spatial domain, and spatial model
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,5,5);
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
    //tProblem.output("cfd_test_problem");

    // call outlet criterion
    auto tTol = 1e-2;
    auto tCriterionValue = tProblem.criterionValue(tControls, "Outlet Average Surface Pressure");
    TEST_FLOATING_EQUALITY(0.0, tCriterionValue, tTol);
    tCriterionValue = tProblem.criterionValue(tControls, "Inlet Average Surface Pressure");
    TEST_FLOATING_EQUALITY(0.0896025, tCriterionValue, tTol);

    std::system("rm -f cfd_solver_diagnostics.txt");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, IsothermalFlowOnChannel_Re100_WithBrinkmanTerm)
{
    // set xml file inputs
    Teuchos::RCP<Teuchos::ParameterList> tInputs =
        Teuchos::getParametersFromXmlString(
            "<ParameterList name='Plato Problem'>"
            "  <ParameterList name='Hyperbolic'>"
            "    <Parameter name='Scenario' type='string' value='Density TO'/>"
            "    <Parameter name='Heat Transfer' type='string' value='None'/>"
            "    <ParameterList  name='Momentum Conservation'>"
            "      <Parameter  name='Stabilization Constant' type='double' value='1.0'/>"
            "    </ParameterList>"
            "    <ParameterList  name='Dimensionless Properties'>"
            "      <Parameter  name='Reynolds Number'  type='double'  value='100'/>"
            "      <Parameter  name='Impermeability Number'  type='double'  value='1e2'/>"
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
            "      <Parameter  name='Value'    type='double' value='1'/>"
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
            "    <Parameter name='Safety Factor' type='double' value='0.1'/>"
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
    //auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(15,1,150,20);
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(10,1,100,10);
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
    Plato::blas1::fill(0.1, tControls);
    auto tSolution = tProblem.solution(tControls);
    //tProblem.output("cfd_test_problem");

    // test solution
    auto tTags = tSolution.tags();
    std::vector<std::string> tGoldTags = { "velocity", "pressure" };
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
    TEST_FLOATING_EQUALITY(367.334, tMaxPress, tTol);
    Plato::Scalar tMinPress = 0;
    Plato::blas1::min(tPressSubView, tMinPress);
    TEST_FLOATING_EQUALITY(0.0, tMinPress, tTol);
    //Plato::print(tPressSubView, "steady state pressure");

    auto tVelocity = tSolution.get("velocity");
    auto tVelSubView = Kokkos::subview(tVelocity, 1, Kokkos::ALL());
    Plato::Scalar tMaxVel = 0;
    Plato::blas1::max(tVelSubView, tMaxVel);
    TEST_FLOATING_EQUALITY(1.0, tMaxVel, tTol);
    Plato::Scalar tMinVel = 0;
    Plato::blas1::min(tVelSubView, tMinVel);
    TEST_FLOATING_EQUALITY(-0.0519869, tMinVel, tTol);
    //Plato::print(tVelSubView, "steady state velocity");
    
    std::system("rm -f cfd_solver_diagnostics.txt");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, IsothermalFlowOnChannel_Re100)
{
    // set xml file inputs
    Teuchos::RCP<Teuchos::ParameterList> tInputs =
        Teuchos::getParametersFromXmlString(
            "<ParameterList name='Plato Problem'>"
            "  <ParameterList name='Hyperbolic'>"
            "    <Parameter name='Heat Transfer' type='string' value='None'/>"
            "    <ParameterList  name='Momentum Conservation'>"
            "      <Parameter  name='Stabilization Constant' type='double' value='1.0'/>"
            "    </ParameterList>"
            "    <ParameterList  name='Dimensionless Properties'>"
            "      <Parameter  name='Reynolds Number'  type='double'  value='100'/>"
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
            "      <Parameter  name='Value'    type='double' value='1'/>"
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
            "    <Parameter name='Safety Factor' type='double' value='0.7'/>"
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
    //auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(10,1,150,20);
    //auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(10,1,100,10);
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,5,5);
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
    //tProblem.output("cfd_test_problem");

    // test solution
    auto tTags = tSolution.tags();
    std::vector<std::string> tGoldTags = { "velocity", "pressure" };
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
    TEST_FLOATING_EQUALITY(0.163373, tMaxPress, tTol);
    Plato::Scalar tMinPress = 0;
    Plato::blas1::min(tPressSubView, tMinPress);
    TEST_FLOATING_EQUALITY(0.0, tMinPress, tTol);
    //Plato::print(tPressSubView, "steady state pressure");

    auto tVelocity = tSolution.get("velocity");
    auto tVelSubView = Kokkos::subview(tVelocity, 1, Kokkos::ALL());
    Plato::Scalar tMaxVel = 0;
    Plato::blas1::max(tVelSubView, tMaxVel);
    TEST_FLOATING_EQUALITY(1.09563, tMaxVel, tTol);
    Plato::Scalar tMinVel = 0;
    Plato::blas1::min(tVelSubView, tMinVel);
    TEST_FLOATING_EQUALITY(-0.0477337, tMinVel, tTol);
    //Plato::print(tVelSubView, "steady state velocity");
    
    std::system("rm -f cfd_solver_diagnostics.txt");
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
            "    <Parameter name='Safety Factor'      type='double' value='0.7'/>"
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
    std::vector<std::string> tGoldTags = { "velocity", "pressure" };
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
    TEST_FLOATING_EQUALITY(0.10276, tMaxPress, tTol);
    Plato::Scalar tMinPress = 0;
    Plato::blas1::min(tPressSubView, tMinPress);
    TEST_FLOATING_EQUALITY(-0.577537, tMinPress, tTol);
    //Plato::print(tPressSubView, "steady state pressure");

    auto tVelocity = tSolution.get("velocity");
    auto tVelSubView = Kokkos::subview(tVelocity, 1, Kokkos::ALL());
    Plato::Scalar tMaxVel = 0;
    Plato::blas1::max(tVelSubView, tMaxVel);
    TEST_FLOATING_EQUALITY(1.0, tMaxVel, tTol);
    Plato::Scalar tMinVel = 0;
    Plato::blas1::min(tVelSubView, tMinVel);
    TEST_FLOATING_EQUALITY(-0.33372, tMinVel, tTol);
    //Plato::print(tVelSubView, "steady state velocity");
    
    std::system("rm -f cfd_solver_diagnostics.txt");
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
            "    <Parameter name='Safety Factor'      type='double' value='0.7'/>"
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

    auto tTol = 1e-2;
    auto tPressure = tSolution.get("pressure");
    auto tPressSubView = Kokkos::subview(tPressure, 1, Kokkos::ALL());
    Plato::Scalar tMaxPress = 0;
    Plato::blas1::max(tPressSubView, tMaxPress);
    TEST_FLOATING_EQUALITY(2.20016, tMaxPress, tTol);
    Plato::Scalar tMinPress = 0;
    Plato::blas1::min(tPressSubView, tMinPress);
    TEST_FLOATING_EQUALITY(0.0, tMinPress, tTol);
    //Plato::print(tPressSubView, "steady state pressure");

    auto tVelocity = tSolution.get("velocity");
    auto tVelSubView = Kokkos::subview(tVelocity, 1, Kokkos::ALL());
    Plato::Scalar tMaxVel = 0;
    Plato::blas1::max(tVelSubView, tMaxVel);
    TEST_FLOATING_EQUALITY(1.0, tMaxVel, tTol);
    Plato::Scalar tMinVel = 0;
    Plato::blas1::min(tVelSubView, tMinVel);
    TEST_FLOATING_EQUALITY(-0.633259, tMinVel, tTol);
    //Plato::print(tVelSubView, "steady state velocity");
    
    std::system("rm -f cfd_solver_diagnostics.txt");
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
            "      <Parameter  name='Prandtl Number'  type='double' value='0.71'/>"
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
            "  <ParameterList name='Material Models'>"
            "    <ParameterList name='Air'>"
            "      <ParameterList name='Thermal Properties'>"
            "        <Parameter  name='Thermal Diffusivity' type='double' value='2.1117e-5'/>"
            "      </ParameterList>"
            "      <ParameterList name='Viscous Properties'>"
            "        <Parameter  name='Kinematic Viscocity' type='double' value='1.5111e-5'/>"
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
            "    <Parameter name='Temperature Tolerance' type='double' value='1e-5'/>"
            "  </ParameterList>"
            "  <ParameterList  name='Time Integration'>"
            "    <Parameter name='Damping' type='double' value='0.1'/>"
            "    <Parameter name='Safety Factor' type='double' value='0.4'/>"
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
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,25,25);
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
    TEST_FLOATING_EQUALITY(252.224, tMaxPress, tTol);
    Plato::Scalar tMinPress = 0;
    Plato::blas1::min(tPressSubView, tMinPress);
    TEST_FLOATING_EQUALITY(-229.947, tMinPress, tTol);
    //Plato::print(tPressSubView, "steady state pressure");

    auto tVelocity = tSolution.get("velocity");
    auto tVelSubView = Kokkos::subview(tVelocity, 1, Kokkos::ALL());
    Plato::Scalar tMaxVel = 0;
    Plato::blas1::max(tVelSubView, tMaxVel);
    TEST_FLOATING_EQUALITY(3.70439, tMaxVel, tTol);
    Plato::Scalar tMinVel = 0;
    Plato::blas1::min(tVelSubView, tMinVel);
    TEST_FLOATING_EQUALITY(-3.34883, tMinVel, tTol);
    //Plato::print(tVelSubView, "steady state velocity");

    auto tTemperature = tSolution.get("temperature");
    auto tTempSubView = Kokkos::subview(tTemperature, 1, Kokkos::ALL());
    auto tTempNorm = Plato::blas1::norm(tTempSubView);
    TEST_FLOATING_EQUALITY(15.077, tTempNorm, tTol);
    //Plato::print(tTempSubView, "steady state temperature");
    
    std::system("rm -f cfd_solver_diagnostics.txt");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, NaturalConvectionSquareEnclosure_Ra1e4)
{
    // set xml file inputs
    Teuchos::RCP<Teuchos::ParameterList> tInputs =
        Teuchos::getParametersFromXmlString(
            "<ParameterList name='Plato Problem'>"
            "  <ParameterList name='Hyperbolic'>"
            "    <Parameter name='Heat Transfer' type='string' value='Natural'/>"
            "    <ParameterList  name='Dimensionless Properties'>"
            "      <Parameter  name='Prandtl Number'  type='double' value='0.71'/>"
            "      <Parameter  name='Rayleigh Number' type='Array(double)' value='{0,1e4}'/>"
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
            "  <ParameterList name='Material Models'>"
            "    <ParameterList name='Air'>"
            "      <ParameterList name='Thermal Properties'>"
            "        <Parameter  name='Thermal Diffusivity' type='double' value='2.1117e-5'/>"
            "      </ParameterList>"
            "      <ParameterList name='Viscous Properties'>"
            "        <Parameter  name='Kinematic Viscocity' type='double' value='1.5111e-5'/>"
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
            "  <ParameterList  name='Time Integration'>"
            "    <Parameter name='Damping' type='double' value='0.3'/>"
            "    <Parameter name='Safety Factor' type='double' value='0.4'/>"
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
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,25,25);
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
    TEST_FLOATING_EQUALITY(4155.81, tMaxPress, tTol);
    Plato::Scalar tMinPress = 0;
    Plato::blas1::min(tPressSubView, tMinPress);
    TEST_FLOATING_EQUALITY(-9.88111, tMinPress, tTol);
    //Plato::print(tPressSubView, "steady state pressure");

    auto tVelocity = tSolution.get("velocity");
    auto tVelSubView = Kokkos::subview(tVelocity, 1, Kokkos::ALL());
    Plato::Scalar tMaxVel = 0;
    Plato::blas1::max(tVelSubView, tMaxVel);
    TEST_FLOATING_EQUALITY(19.4625, tMaxVel, tTol);
    Plato::Scalar tMinVel = 0;
    Plato::blas1::min(tVelSubView, tMinVel);
    TEST_FLOATING_EQUALITY(-16.1093, tMinVel, tTol);
    //Plato::print(tVelSubView, "steady state velocity");

    auto tTemperature = tSolution.get("temperature");
    auto tTempSubView = Kokkos::subview(tTemperature, 1, Kokkos::ALL());
    auto tTempNorm = Plato::blas1::norm(tTempSubView);
    TEST_FLOATING_EQUALITY(14.1776, tTempNorm, tTol);
    //Plato::print(tTempSubView, "steady state temperature");
    
    std::system("rm -f cfd_solver_diagnostics.txt");
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
    TEST_FLOATING_EQUALITY(4.21189, tValue, tTol);
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
    auto tTol = 1e-4;
    TEST_FLOATING_EQUALITY(3.2, tValue, tTol);
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
    auto tTol = 1e-4;
    auto tHostResidual = Kokkos::create_mirror(tResidual);
    Kokkos::deep_copy(tHostResidual, tResidual);
    std::vector<Plato::Scalar> tGold = {0.5,1.4,2.3,3.2};
    for (Plato::OrdinalType tNode = 0; tNode < tNumNodes; tNode++)
    {
        TEST_FLOATING_EQUALITY(tGold[tNode], tHostResidual(tNode), tTol); // @suppress("Invalid arguments")
    }
    //Plato::print(tResidual, "residual");
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

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PressureIncrementResidual_EvaluateBoundary)
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
    auto tTol = 1e-4;
    auto tHostResult = Kokkos::create_mirror(tResult);
    Kokkos::deep_copy(tHostResult, tResult);
    std::vector<std::vector<Plato::Scalar>> tGold =
        {{0.0,0.0,0.0},{0.0,-217.0,-217.0}};
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

    // evaluate pressure increment residual
    Plato::DataMap tDataMap;
    Plato::ScalarMultiVectorT<EvaluationT::ResultScalarType> tResult("result", tNumCells, PhysicsT::mNumMassDofsPerCell);
    Plato::Fluids::PressureResidual<PhysicsT,EvaluationT> tResidual(tDomain,tDataMap);
    tResidual.evaluate(tWorkSets, tResult);

    // test values
    auto tTol = 1e-4;
    auto tHostResult = Kokkos::create_mirror(tResult);
    Kokkos::deep_copy(tHostResult, tResult);
    std::vector<std::vector<Plato::Scalar>> tGold =
        {{4.5,1.66667,-6.16667},{-15.5,-1.66667,17.1667}};
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

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, IntegrateDivergenceOperator)
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
    std::vector<std::vector<Plato::Scalar>> tGold = {{-0.166667,-0.166667,0.333333},{0.5,0.166667,-0.666667}};
    for (Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++)
    {
        for (Plato::OrdinalType tDof = 0; tDof < tNumNodesPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(tGold[tCell][tDof], tHostResult(tCell, tDof), tTol);
        }
    }
    //Plato::print_array_2D(tResult, "results");
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
        Plato::Fluids::calculate_flux_divergence<tNumNodesPerCell,tSpaceDims>(aCellOrdinal, tGradient, tCellVolume, tFlux, tResult, 1.0);
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
        Plato::Fluids::integrate_scalar_field<tNumNodesPerCell>(aCellOrdinal, tBasisFunctions, tCellVolume, tSource, tResult, 1.0);
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
    std::vector<Plato::Scalar> tGold = {-2.43333,-2.77778,-1.48333,-1.65,-2.43333,-2.77778,-0.95,-1.1277};
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

    auto tTol = 1e-4;
    std::vector<std::vector<Plato::Scalar>> tGold =
        {{-0.5,-0.5,-0.5,-0.5,1.0,1.0}, {1.5,1.5,0.5,0.5,-2.0,-2.0}};
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
    std::vector<std::vector<Plato::Scalar>> tGold = {{14.0,14.0},{-38.0,-38.0}};
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

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, BLAS2_update)
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
        Plato::blas2::update<tNumDofsPerCell>(aCellOrdinal, 2.0, tVec1, 3.0 + tConstant, tVec2);
    },"device_blas2_update");

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
    auto tMyNodeSetOrdinals = Plato::omega_h::get_entity_ordinals<Omega_h::NODE_SET>(tMeshSets, "x+");
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
    auto tMySideSetOrdinals = Plato::omega_h::get_entity_ordinals<Omega_h::SIDE_SET>(tMeshSets, "x+");
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
    TEST_EQUALITY(true, Plato::omega_h::is_entity_set_defined<Omega_h::NODE_SET>(tMeshSets, "x+"));
    TEST_EQUALITY(false, Plato::omega_h::is_entity_set_defined<Omega_h::NODE_SET>(tMeshSets, "dog"));

    TEST_EQUALITY(true, Plato::omega_h::is_entity_set_defined<Omega_h::SIDE_SET>(tMeshSets, "x+"));
    TEST_EQUALITY(false, Plato::omega_h::is_entity_set_defined<Omega_h::SIDE_SET>(tMeshSets, "dog"));
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
        {-0.318111, -0.379111, -0.191667, -0.225, -0.318111, -0.329111, -0.126444, -0.104111};
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
    auto tNumEntities = Plato::omega_h::get_num_entities(Omega_h::VERT, tMesh.operator*());
    TEST_EQUALITY(4, tNumEntities);
    tNumEntities = Plato::omega_h::get_num_entities(Omega_h::EDGE, tMesh.operator*());
    TEST_EQUALITY(5, tNumEntities);
    tNumEntities = Plato::omega_h::get_num_entities(Omega_h::FACE, tMesh.operator*());
    TEST_EQUALITY(2, tNumEntities);
    tNumEntities = Plato::omega_h::get_num_entities(Omega_h::REGION, tMesh.operator*());
    TEST_EQUALITY(2, tNumEntities);
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

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, BLAS2_DeviceScale)
{
    constexpr Plato::OrdinalType tNumCells = 2;
    constexpr Plato::OrdinalType tNumSpaceDims = 2;
    Plato::ScalarMultiVector tInput("input", tNumCells, tNumSpaceDims);
    Plato::blas2::fill(1.0, tInput);
    Plato::ScalarMultiVector tOutput("output", tNumCells, tNumSpaceDims);

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        Plato::blas2::scale<tNumSpaceDims>(aCellOrdinal, 4.0, tInput, tOutput);
    }, "device blas2::scale");

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
        Plato::blas2::scale<tNumSpaceDims>(aCellOrdinal, 4.0, tInput);
    }, "device blas2::scale");

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
        Plato::blas2::dot<tNumSpaceDims>(aCellOrdinal, tInputA, tInputB, tOutput);
    }, "device blas2::dot");

    auto tTol = 1e-6;
    auto tHostOutput = Kokkos::create_mirror(tOutput);
    Kokkos::deep_copy(tHostOutput, tOutput);
    for (Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++)
    {
        TEST_FLOATING_EQUALITY(8.0, tHostOutput(tCell), tTol);
    }
}


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, BLAS3_DeviceScale)
{
    constexpr Plato::OrdinalType tNumCells = 2;
    constexpr Plato::OrdinalType tNumSpaceDims = 2;
    Plato::ScalarArray3D tInput("input", tNumCells, tNumSpaceDims, tNumSpaceDims);
    Plato::blas3::fill<tNumSpaceDims, tNumSpaceDims>(tNumCells, 1.0, tInput);
    Plato::ScalarArray3D tOutput("output", tNumCells, tNumSpaceDims, tNumSpaceDims);

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        Plato::blas3::scale<tNumSpaceDims, tNumSpaceDims>(aCellOrdinal, 4.0, tInput, tOutput);
    }, "device blas3::scale");

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

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, BLAS3_Dot)
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
        Plato::blas3::dot<tNumSpaceDims, tNumSpaceDims>(aCellOrdinal, tInputA, tInputB, tOutput);
    }, "device blas3::dot");

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
    auto tNames = Plato::teuchos::parse_array<std::string>("Functions", tParams.operator*());

    std::vector<std::string> tGoldNames = {"My Inlet Pressure", "My Outlet Pressure"};
    for(auto& tName : tNames)
    {
        auto tIndex = &tName - &tNames[0];
        TEST_EQUALITY(tGoldNames[tIndex], tName);
    }

    auto tWeights = Plato::teuchos::parse_array<Plato::Scalar>("Weights", *tParams);
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
    auto tScalarOutput = Plato::teuchos::parse_parameter<Plato::Scalar>("Prandtl", "Dimensionless Properties", tParams.operator*());
    auto tTolerance = 1e-6;
    TEST_FLOATING_EQUALITY(tScalarOutput, 2.1, tTolerance);

    // Darcy #
    tScalarOutput = Plato::teuchos::parse_parameter<Plato::Scalar>("Darcy", "Dimensionless Properties", tParams.operator*());
    TEST_FLOATING_EQUALITY(tScalarOutput, 2.2, tTolerance);

    // Grashof #
    auto tArrayOutput = Plato::teuchos::parse_parameter<Teuchos::Array<Plato::Scalar>>("Grashof", "Dimensionless Properties", tParams.operator*());
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
