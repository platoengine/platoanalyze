/*
 * AverageSurfacePressure.hpp
 *
 *  Created on: Apr 6, 2021
 */

#pragma once

#include "SpatialModel.hpp"
#include "UtilsTeuchos.hpp"
#include "SurfaceIntegralUtilities.hpp"
#include "LinearTetCubRuleDegreeOne.hpp"

#include "hyperbolic/AbstractScalarFunction.hpp"

namespace Plato
{

namespace Fluids
{

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
            auto tFaceOrdinalsOnSideSet = Plato::omega_h::side_set_face_ordinals(mSpatialDomain.MeshSets, tName);
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

}
// namespace Fluids

}
// namespace Plato
