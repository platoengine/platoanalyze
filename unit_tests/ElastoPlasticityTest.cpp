/*
 * ElastoPlasticityTest.cpp
 *
 *  Created on: Sep 30, 2019
 */

#include <memory>

#include <Omega_h_mesh.hpp>
#include <Omega_h_assoc.hpp>

#include "plato/Simplex.hpp"
#include "plato/BodyLoads.hpp"
#include "plato/NaturalBCs.hpp"
#include "plato/ScalarGrad.hpp"
#include "plato/SimplexFadTypes.hpp"
#include "plato/PlatoStaticsTypes.hpp"
#include "plato/LinearElasticMaterial.hpp"
#include "plato/ThermoPlasticityUtilities.hpp"
#include "plato/LinearTetCubRuleDegreeOne.hpp"

namespace Plato
{

/******************************************************************************//**
 * \brief Abstract vector function interface for Variational Multi-Scale (VMS)
 *   Partial Differential Equations (PDEs) with history dependent states
 * \tparam EvaluationType evaluation type use to determine automatic differentiation
 *   type for the vector function (e.g. Residual, Jacobian, GradientZ, GradientU, etc.)
 **********************************************************************************/
template<typename EvaluationType>
class AbstractVectorFunctionVMSInc
{
protected:
    Omega_h::Mesh& mMesh;
    Plato::DataMap& mDataMap;
    Omega_h::MeshSets& mMeshSets;
    std::vector<std::string> mDofNames;

public:
    /**************************************************************************//**
     * \brief Constructor
     * \param [in]  aMesh mesh metadata
     * \param [in]  aMeshSets mesh side-sets metadata
     * \param [in]  aDataMap output data map
     ******************************************************************************/
    explicit AbstractVectorFunctionVMSInc(Omega_h::Mesh& aMesh, Omega_h::MeshSets& aMeshSets, Plato::DataMap& aDataMap) :
            mMesh(aMesh),
            mDataMap(aDataMap),
            mMeshSets(aMeshSets)
    {
    }

    /**************************************************************************//**
     * \brief Destructor
     ******************************************************************************/
    virtual ~AbstractVectorFunctionVMSInc()
    {
    }

    /****************************************************************************//**
     * \brief Return reference to Omega_h mesh data base
     * \return mesh metadata
     ********************************************************************************/
    decltype(mMesh) getMesh() const
    {
        return (mMesh);
    }

    /****************************************************************************//**
     * \brief Return reference to Omega_h mesh sets
     * \return mesh side sets metadata
     ********************************************************************************/
    decltype(mMeshSets) getMeshSets() const
    {
        return (mMeshSets);
    }

    /****************************************************************************//**
     * \brief Evaluate the stabilized residual equation
     * \param [in] aGlobalState current global state ( i.e. state at the n-th time interval (\f$ t^{n} \f$) )
     * \param [in] aGlobalStatePrev previous global state ( i.e. state at the n-th minus one time interval (\f$ t^{n-1} \f$) )
     * \param [in] aLocalState current local state ( i.e. state at the n-th time interval (\f$ t^{n} \f$) )
     * \param [in] aLocalStatePrev previous local state ( i.e. state at the n-th minus one time interval (\f$ t^{n-1} \f$) )
     * \param [in] aPressureGrad current pressure gradient ( i.e. state at the n-th time interval (\f$ t^{n} \f$) )
     * \param [in] aControl design variables
     * \param [in/out] aResult residual evaluation
     * \param [in] aTimeStep current time step (i.e. \f$ \Delta{t}^{n} \f$), default = 0.0
    ********************************************************************************/
    virtual void
    evaluate(const Plato::ScalarMultiVectorT<typename EvaluationType::StateScalarType> & aGlobalState,
             const Plato::ScalarMultiVectorT<typename EvaluationType::PrevStateScalarType> & aGlobalStatePrev,
             const Plato::ScalarMultiVectorT<typename EvaluationType::LocalStateScalarType> & aLocalState,
             const Plato::ScalarMultiVectorT<typename EvaluationType::PrevLocalStateScalarType> & aLocalStatePrev,
             const Plato::ScalarMultiVectorT<typename EvaluationType::NodeStateScalarType> & aPressureGrad,
             const Plato::ScalarMultiVectorT<typename EvaluationType::ControlScalarType> & aControl,
             const Plato::ScalarArray3DT<typename EvaluationType::ConfigScalarType> & aConfig,
             const Plato::ScalarMultiVectorT<typename EvaluationType::ResultScalarType> & aResult,
             Plato::Scalar aTimeStep = 0.0) const = 0;
};
// class AbstractVectorFunctionVMSInc

template<Plato::OrdinalType SpaceDim, Plato::OrdinalType NumControls = 1>
class SimplexStabilizedElastoPasticity : public Plato::Simplex<SpaceDim>
{
public:
    using Plato::Simplex<SpaceDim>::mNumNodesPerCell; /*!< number of spatial dimensions */
    using Plato::Simplex<SpaceDim>::mNumSpatialDims; /*!< number of nodes per cell */

    static constexpr Plato::OrdinalType mNumVoigtTerms =
            (SpaceDim == 3) ? 6 : ((SpaceDim == 2) ? 3 : (((SpaceDim == 1) ? 1 : 0))); /*!< number of Voigt terms */

    // degree-of-freedom attributes
    //
    static constexpr Plato::OrdinalType mPDofOffset = SpaceDim; /*!< number of pressure degrees of freedom offset */
    static constexpr Plato::OrdinalType mNumDofsPerNode = SpaceDim + 1; /*!< number of degrees of freedom per node { disp_x, disp_y, disp_z, pressure} */
    static constexpr Plato::OrdinalType mNumDofsPerCell = mNumDofsPerNode * mNumNodesPerCell; /*!< number of degrees of freedom per cell */
    static constexpr Plato::OrdinalType mNumControl = NumControls; /*!< number of controls */

    // this physics can be used with VMS functionality in PA.  The
    // following defines the nodal state attributes required by VMS
    //
    static constexpr Plato::OrdinalType mNumNSPerNode = SpaceDim; /*!< number of node states per node */
    static constexpr Plato::OrdinalType mNumNSPerCell = mNumNSPerNode * mNumNodesPerCell; /*!< number of node states per cell */

    static constexpr Plato::OrdinalType mNumLocalDofsPerCell =
            (SpaceDim == 3) ? 14 : ((SpaceDim == 2) ? 8 : (((SpaceDim == 1) ? 4 : 0))); /*!< number of local degrees of freedom per cell for J2-plasticity*/
};
// class SimplexStabilizedElastoPasticity













/**************************************************************************//**
 * \brief J2 Plasticity Local Residual class
 ******************************************************************************/
template<typename EvaluationType, typename PhysicsType>
class StabilizedElastoPlasticityResidual : public Plato::AbstractVectorFunctionVMSInc<EvaluationType>
{
// Private member data
private:
    static constexpr Plato::OrdinalType mSpaceDim = EvaluationType::SpatialDim; /*!< spatial dimensions */

    using Plato::Simplex<mSpaceDim>::mNumNodesPerCell; /*!< number of nodes per cell */
    using Plato::SimplexStabilizedElastoPasticity<mSpaceDim>::mNumVoigtTerms; /*!< number of Voigt terms */
    using Plato::SimplexStabilizedElastoPasticity<mSpaceDim>::mNumDofsPerNode; /*!< number of nodes per node */
    using Plato::SimplexStabilizedElastoPasticity<mSpaceDim>::mNumDofsPerCell; /*!< number of nodes per cell */

    static constexpr Plato::OrdinalType mNumMechDims = mSpaceDim; /*!< number of mechanical degrees of freedom */
    static constexpr Plato::OrdinalType mMechDofOffset = 0; /*!< mechanical degrees of freedom offset */
    static constexpr Plato::OrdinalType mPressDofOffset = mSpaceDim; /*!< pressure degree of freedom offset */

    static constexpr Plato::OrdinalType mNumNodesPerCell = PhysicsType::mNumNodesPerCell; /*!< number nodes per cell */
    static constexpr Plato::OrdinalType mNumVoigtTerms = PhysicsType::mNumVoigtTerms; /*!< number of voigt terms */

    using Plato::AbstractVectorFunctionVMSInc<EvaluationType>::mMesh; /*!< mesh database */
    using Plato::AbstractVectorFunctionVMSInc<EvaluationType>::mDataMap; /*!< PLATO Engine output database */

    using GlobalStateT = typename EvaluationType::StateScalarType; /*!< global state variables automatic differentiation type */
    using PrevGlobalStateT = typename EvaluationType::PrevStateScalarType; /*!< global state variables automatic differentiation type */
    using LocalStateT = typename EvaluationType::LocalStateScalarType; /*!< local state variables automatic differentiation type */
    using PrevLocalStateT = typename EvaluationType::PrevLocalStateScalarType; /*!< local state variables automatic differentiation type */
    using NodeStateT = typename EvaluationType::NodeStateScalarType; /*!< Node State AD type */
    using ControlT = typename EvaluationType::ControlScalarType; /*!< control variables automatic differentiation type */
    using ConfigT = typename EvaluationType::ConfigScalarType; /*!< config variables automatic differentiation type */
    using ResultT = typename EvaluationType::ResultScalarType; /*!< result variables automatic differentiation type */

    std::vector<std::string> mPlotTable; /*!< array with output data identifiers */

    std::shared_ptr<Plato::BodyLoads<EvaluationType>> mBodyLoads; /*!< body loads interface */
    std::shared_ptr<Plato::NaturalBCs<mSpaceDim, mNumMechDims, mNumDofsPerNode, mMechDofOffset>> mBoundaryLoads; /*!< boundary loads interface */

    Teuchos::RCP<Plato::LinearElasticMaterial<mSpaceDim>> mMaterialModel; /*!< material model interface */
    std::shared_ptr<Plato::LinearTetCubRuleDegreeOne<mSpaceDim>> mCubatureRule; /*!< linear cubature rule */

// Private access functions
private:
    /******************************************************************************//**
     * \brief initialize material, loads and output data
     * \param [in] aProblemParams input XML data
     **********************************************************************************/
    void initialize(Teuchos::ParameterList& aProblemParams)
    {
        // create material model and get stiffness
        //
        Plato::ElasticModelFactory<mSpaceDim> tMaterialFactory(aProblemParams);
        mMaterialModel = tMaterialFactory.create();

        // parse body loads
        //
        if(aProblemParams.isSublist("Body Loads"))
        {
            mBodyLoads = std::make_shared<Plato::BodyLoads<EvaluationType>>(aProblemParams.sublist("Body Loads"));
        }

        // parse mechanical boundary Conditions
        //
        if(aProblemParams.isSublist("Mechanical Natural Boundary Conditions"))
        {
            mBoundaryLoads =
                    std::make_shared<Plato::NaturalBCs<mSpaceDim, mNumMechDims, mNumDofsPerNode, mMechDofOffset>>(aProblemParams.sublist("Mechanical Natural Boundary Conditions"));
        }

        auto tResidualParams = aProblemParams.sublist("Elliptic");
        if(tResidualParams.isType < Teuchos::Array < std::string >> ("Plottable"))
        {
            mPlotTable = tResidualParams.get < Teuchos::Array < std::string >> ("Plottable").toVector();
        }
    }

// Public access functions
public:
    /******************************************************************************//**
     * \brief Constructor
     * \param [in] aMesh mesh metadata
     * \param [in] aMeshSets side-sets metadata
     * \param [in] aDataMap output data map
     * \param [in] aProblemParams input XML data
     * \param [in] aPenaltyParams penalty function input XML data
     **********************************************************************************/
    StabilizedElastoPlasticityResidual(Omega_h::Mesh& aMesh,
                                       Omega_h::MeshSets& aMeshSets,
                                       Plato::DataMap& aDataMap,
                                       Teuchos::ParameterList& aProblemParams) :
            Plato::AbstractVectorFunctionVMSInc<EvaluationType>(aMesh, aMeshSets, aDataMap),
            mBodyLoads(nullptr),
            mBoundaryLoads(nullptr),
            mCubatureRule(std::make_shared<Plato::LinearTetCubRuleDegreeOne<mSpatialDim>>())
    {
        this->initialize(aProblemParams);
    }

    /******************************************************************************//**
     * \brief Destructor
     **********************************************************************************/
    virtual ~StabilizedElastoPlasticityResidual(){}

    /****************************************************************************//**
     * \brief Evaluate the stabilized residual equation
     * \param [in] aPressureGrad current pressure gradient ( i.e. state at the n-th time interval (\f$ t^{n} \f$) )
     * \param [in] aGlobalState current global state ( i.e. state at the n-th time interval (\f$ t^{n} \f$) )
     * \param [in] aGlobalStatePrev previous global state ( i.e. state at the n-th minus one time interval (\f$ t^{n-1} \f$) )
     * \param [in] aLocalState current local state ( i.e. state at the n-th time interval (\f$ t^{n} \f$) )
     * \param [in] aLocalStatePrev previous local state ( i.e. state at the n-th minus one time interval (\f$ t^{n-1} \f$) )
     * \param [in] aControl design variables
     * \param [in/out] aResult residual evaluation
     * \param [in] aTimeStep current time step (i.e. \f$ \Delta{t}^{n} \f$), default = 0.0
    ********************************************************************************/
    void evaluate(const Plato::ScalarMultiVectorT<GlobalStateT> & aGlobalState,
                  const Plato::ScalarMultiVectorT<PrevGlobalStateT> & aGlobalStatePrev,
                  const Plato::ScalarMultiVectorT<LocalStateT> & aLocalState,
                  const Plato::ScalarMultiVectorT<PrevLocalStateT> & aLocalStatePrev,
                  const Plato::ScalarMultiVectorT<NodeStateT> & aPressureGrad,
                  const Plato::ScalarMultiVectorT<ControlT> & aControl,
                  const Plato::ScalarArray3DT<ConfigT> & aConfig,
                  const Plato::ScalarMultiVectorT<ResultT> & aResult,
                  Plato::Scalar aTimeStep = 0.0)
    {
        auto tNumCells = mMesh.nelems();

        using GradScalarT = typename Plato::fad_type_t<PhysicsType, GlobalStateT, ConfigT>;
        using ElasticStrainT = typename Plato::fad_type_t<PhysicsType, LocalStateT, ConfigT, GlobalStateT>;

        Plato::ScalarGrad<mSpaceDim> tComputeScalarGrad;
        Plato::ComputeGradientWorkset <mSpaceDim> tComputeGradient;
        Plato::ThermoPlasticityUtilities<EvaluationType::SpatialDim, PhysicsType> tThermoPlasticityUtils;

        Plato::ScalarVectorT<ConfigT> tCellVolume("cell volume", tNumCells);
        Plato::ScalarMultiVectorT <GradScalarT> tPressGradL2 ("pressure grad on L2", tNumCells, mSpaceDim);
        Plato::ScalarMultiVectorT<ElasticStrainT> tElasticStrain("elastic strain", tNumCells,mNumVoigtTerms);
        Plato::ScalarArray3DT<ConfigT> tGradient(" configuration gradient", tNumCells, mNumNodesPerCell, mSpaceDim);

        auto tQuadratureWeight = mCubatureRule->getCubWeight();
        auto tBasisFunctions   = mCubatureRule->getBasisFunctions();

        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
        {
            // compute configuration gradients
            tComputeGradient(aCellOrdinal, tGradient, aConfig, tCellVolume);
            tCellVolume(aCellOrdinal) *= tQuadratureWeight;

            // compute elastic strain
            tThermoPlasticityUtils.computeElasticStrain(aCellOrdinal, aGlobalState, aLocalState, tBasisFunctions, tGradient, tElasticStrain);

            // compute pressure gradient
            tComputeScalarGrad(aCellOrdinal, tPressGradL2);
        }, "stabilized elasto-plastic residual evaluation");
    }
};
// class J2PlasticityLocalResidual

}
// namespace Plato

namespace ElastoPlasticityTest
{

}
// ElastoPlasticityTest
