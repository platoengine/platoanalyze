/*
 * ElastoPlasticityTest.cpp
 *
 *  Created on: Sep 30, 2019
 */

#include <memory>

#include <Omega_h_mesh.hpp>
#include <Omega_h_assoc.hpp>

#include "plato/Simp.hpp"
#include "plato/Simplex.hpp"
#include "plato/Kinetics.hpp"
#include "plato/BodyLoads.hpp"
#include "plato/NaturalBCs.hpp"
#include "plato/ScalarGrad.hpp"
#include "plato/WorksetBase.hpp"
#include "plato/ProjectToNode.hpp"
#include "plato/FluxDivergence.hpp"
#include "plato/SimplexFadTypes.hpp"
#include "plato/StressDivergence.hpp"
#include "plato/VectorFunctionVMS.hpp"
#include "plato/PlatoStaticsTypes.hpp"
#include "plato/PressureDivergence.hpp"
#include "plato/StabilizedMechanics.hpp"
#include "plato/PlatoAbstractProblem.hpp"
#include "plato/Plato_TopOptFunctors.hpp"
#include "plato/InterpolateFromNodal.hpp"
#include "plato/LinearElasticMaterial.hpp"
#include "plato/LocalVectorFunctionInc.hpp"
#include "plato/ThermoPlasticityUtilities.hpp"
#include "plato/LinearTetCubRuleDegreeOne.hpp"

namespace Plato
{

template<Plato::OrdinalType SpaceDim, Plato::OrdinalType NumControls = 1>
class SimplexElastoPasticity: public Plato::Simplex<SpaceDim>
{
public:
    using Plato::Simplex<SpaceDim>::mNumSpatialDims;  /*!< number of nodes per cell */
    using Plato::Simplex<SpaceDim>::mNumNodesPerCell; /*!< number of spatial dimensions */

    static constexpr Plato::OrdinalType mNumVoigtTerms =
            (SpaceDim == 3) ? 6 : ((SpaceDim == 2) ? 3 : (((SpaceDim == 1) ? 1 : 0))); /*!< number of Voigt terms */

    // degree-of-freedom attributes
    static constexpr auto mNumControl = NumControls;                            /*!< number of controls */
    static constexpr auto mNumDofsPerNode = SpaceDim + 1;                       /*!< number of degrees of freedom per node { disp_x, disp_y, disp_z, pressure} */
    static constexpr auto mPressureDofOffset = SpaceDim;                        /*!< number of pressure degrees of freedom offset */
    static constexpr auto mNumDofsPerCell = mNumDofsPerNode * mNumNodesPerCell; /*!< number of degrees of freedom per cell */

    // this physics can be used with VMS functionality in PA.  The
    // following defines the nodal state attributes required by VMS
    static constexpr auto mNumNodeStatePerNode = SpaceDim;                                /*!< number of node states, i.e. pressure gradient, dofs per node */
    static constexpr auto mNumNodeStatePerCell = mNumNodeStatePerNode * mNumNodesPerCell; /*!< number of node states, i.e. pressure gradient, dofs  per cell */

    static constexpr Plato::OrdinalType mNumLocalDofsPerCell =
            (SpaceDim == 3) ? 14 : ((SpaceDim == 2) ? 8 : (((SpaceDim == 1) ? 4 : 0))); /*!< number of local degrees of freedom per cell for J2-plasticity*/
};
// class SimplexElastoPasticity







/******************************************************************************//**
 * \brief Abstract vector function interface for Variational Multi-Scale (VMS)
 *   Partial Differential Equations (PDEs) with history dependent states
 * \tparam EvaluationType evaluation type use to determine automatic differentiation
 *   type for the vector function (e.g. Residual, Jacobian, GradientZ, GradientU, etc.)
 **********************************************************************************/
template<typename EvaluationType>
class AbstractVectorFunctionVMSInc
{
// Protected member data
protected:
    Omega_h::Mesh &mMesh;
    Plato::DataMap &mDataMap;
    Omega_h::MeshSets &mMeshSets;
    std::vector<std::string> mDofNames;

// Public access functions
public:
    /**************************************************************************//**
     * \brief Constructor
     * \param [in]  aMesh mesh metadata
     * \param [in]  aMeshSets mesh side-sets metadata
     * \param [in]  aDataMap output data map
     ******************************************************************************/
    explicit AbstractVectorFunctionVMSInc(Omega_h::Mesh &aMesh,
                                               Omega_h::MeshSets &aMeshSets,
                                               Plato::DataMap &aDataMap) :
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
    evaluate(const Plato::ScalarMultiVectorT<typename EvaluationType::StateScalarType> &aGlobalState,
             const Plato::ScalarMultiVectorT<typename EvaluationType::PrevStateScalarType> &aGlobalStatePrev,
             const Plato::ScalarMultiVectorT<typename EvaluationType::LocalStateScalarType> &aLocalState,
             const Plato::ScalarMultiVectorT<typename EvaluationType::PrevLocalStateScalarType> &aLocalStatePrev,
             const Plato::ScalarMultiVectorT<typename EvaluationType::NodeStateScalarType> &aPressureGrad,
             const Plato::ScalarMultiVectorT<typename EvaluationType::ControlScalarType> &aControl,
             const Plato::ScalarArray3DT<typename EvaluationType::ConfigScalarType> &aConfig,
             const Plato::ScalarMultiVectorT<typename EvaluationType::ResultScalarType> &aResult,
             Plato::Scalar aTimeStep = 0.0) const = 0;
};
// class AbstractVectorFunctionVMSInc






template<Plato::OrdinalType Length, typename ControlType, typename ResultType>
DEVICE_TYPE inline void
apply_penalty(const Plato::OrdinalType aCellOrdinal, const ControlType & aPenalty, const Plato::ScalarMultiVectorT<ResultType> & aOutput)
{
    for(Plato::OrdinalType tIndex = 0; tIndex < Length; tIndex++)
    {
        aOutput(aCellOrdinal, tIndex) *= aPenalty;
    }
}

template<typename ScalarType>
inline ScalarType compute_shear_modulus(const ScalarType & aElasticModulus, const ScalarType & aPoissonRatio)
{
    ScalarType tShearModulus = aElasticModulus /
        ( static_cast<Plato::Scalar>(2) * ( static_cast<Plato::Scalar>(1) + aPoissonRatio) ) ;
    return (tShearModulus);
}

template<typename ScalarType>
inline ScalarType compute_bulk_modulus(const ScalarType & aElasticModulus, const ScalarType & aPoissonRatio)
{
    ScalarType tShearModulus = aElasticModulus /
        ( static_cast<Plato::Scalar>(3) * ( static_cast<Plato::Scalar>(1) - ( static_cast<Plato::Scalar>(2) * aPoissonRatio) ) );
    return (tShearModulus);
}

Plato::Scalar parse_elastic_modulus(Teuchos::ParameterList & aParamList)
{
    if (aParamList.isParameter("Youngs Modulus"))
    {
        Plato::Scalar tElasticModulus = aParamList.get < Plato::Scalar > ("Youngs Modulus");
        return (tElasticModulus);
    }
    else
    {
        THROWERR("Youngs Modulus parameter is not defined in 'Isotropic Linear Elastic' sublist.")
    }
}

Plato::Scalar parse_poissons_ratio(Teuchos::ParameterList & aParamList)
{
    if (aParamList.isParameter("Poissons Ratio"))
    {
        Plato::Scalar tPoissonsRatio = aParamList.get < Plato::Scalar > ("Poissons Ratio");
        return (tPoissonsRatio);
    }
    else
    {
        THROWERR("Poisson's ratio parameter is not defined in 'Isotropic Linear Elastic' sublist.")
    }
}





template<Plato::OrdinalType SpaceDim>
class DisplacementDivergence
{
public:
    template<typename ResultType, typename StrainType>
    DEVICE_TYPE inline ResultType
    operator()(const Plato::OrdinalType & aCellOrdinal, const Plato::ScalarMultiVectorT<StrainType> & aStrain);
};

template<>
template<typename ResultType, typename StrainType>
DEVICE_TYPE inline ResultType
DisplacementDivergence<3>::operator()(const Plato::OrdinalType & aCellOrdinal, const Plato::ScalarMultiVectorT<StrainType> & aStrain)
{
    ResultType tOutput = aStrain(aCellOrdinal, 0) + aStrain(aCellOrdinal, 1) + aStrain(aCellOrdinal, 2);
    return (tOutput);
}

template<>
template<typename ResultType, typename StrainType>
DEVICE_TYPE inline ResultType
DisplacementDivergence<2>::operator()(const Plato::OrdinalType & aCellOrdinal, const Plato::ScalarMultiVectorT<StrainType> & aStrain)
{
    ResultType tOutput = aStrain(aCellOrdinal, 0) + aStrain(aCellOrdinal, 1);
    return (tOutput);
}

template<>
template<typename ResultType, typename StrainType>
DEVICE_TYPE inline ResultType
DisplacementDivergence<1>::operator()(const Plato::OrdinalType & aCellOrdinal, const Plato::ScalarMultiVectorT<StrainType> & aStrain)
{
    ResultType tOutput = aStrain(aCellOrdinal, 0);
    return (tOutput);
}





template<Plato::OrdinalType SpaceDim>
class ComputeStabilization
{
private:
    Plato::Scalar mTwoOverThree;
    Plato::Scalar mPressureScaling;
    Plato::Scalar mElasticShearModulus;

public:
    explicit ComputeStabilization(const Plato::Scalar & aStabilization, const Plato::Scalar & aShearModulus) :
        mTwoOverThree(2.0/3.0),
        mPressureScaling(aStabilization),
        mElasticShearModulus(aShearModulus)
    {
    }

    ~ComputeStabilization()
    {
    }

    template<typename ConfigT, typename PressGradT, typename ProjPressGradT, typename ResultT>
    DEVICE_TYPE inline void operator()(const Plato::OrdinalType &aCellOrdinal,
                                       const Plato::ScalarVectorT<ConfigT> & aCellVolume,
                                       const Plato::ScalarMultiVectorT<PressGradT> &aPressureGrad,
                                       const Plato::ScalarMultiVectorT<ProjPressGradT> &aProjectedPressureGrad,
                                       const Plato::ScalarMultiVectorT<ResultT> &aStabilization);
};

template<>
template<typename ConfigT, typename PressGradT, typename ProjPressGradT, typename ResultT>
DEVICE_TYPE inline void
ComputeStabilization<3>::operator()(const Plato::OrdinalType & aCellOrdinal,
                                    const Plato::ScalarVectorT<ConfigT> & aCellVolume,
                                    const Plato::ScalarMultiVectorT<PressGradT> & aPressureGrad,
                                    const Plato::ScalarMultiVectorT<ProjPressGradT> & aProjectedPressureGrad,
                                    const Plato::ScalarMultiVectorT<ResultT> & aStabilization)
{
    ConfigT tTau = pow(aCellVolume(aCellOrdinal), mTwoOverThree) / (static_cast<Plato::Scalar>(2.0) * mElasticShearModulus);
    aStabilization(aCellOrdinal, 0) = mPressureScaling * tTau
        * (mPressureScaling * aPressureGrad(aCellOrdinal, 0) - aProjectedPressureGrad(aCellOrdinal, 0));

    aStabilization(aCellOrdinal, 1) = mPressureScaling * tTau
        * (mPressureScaling * aPressureGrad(aCellOrdinal, 1) - aProjectedPressureGrad(aCellOrdinal, 1));

    aStabilization(aCellOrdinal, 2) = mPressureScaling * tTau
        * (mPressureScaling * aPressureGrad(aCellOrdinal, 2) - aProjectedPressureGrad(aCellOrdinal, 2));
}

template<>
template<typename ConfigT, typename PressGradT, typename ProjPressGradT, typename ResultT>
DEVICE_TYPE inline void
ComputeStabilization<2>::operator()(const Plato::OrdinalType & aCellOrdinal,
                                    const Plato::ScalarVectorT<ConfigT> & aCellVolume,
                                    const Plato::ScalarMultiVectorT<PressGradT> & aPressureGrad,
                                    const Plato::ScalarMultiVectorT<ProjPressGradT> & aProjectedPressureGrad,
                                    const Plato::ScalarMultiVectorT<ResultT> & aStabilization)
{
    ConfigT tTau = pow(aCellVolume(aCellOrdinal), mTwoOverThree) / (static_cast<Plato::Scalar>(2.0) * mElasticShearModulus);
    aStabilization(aCellOrdinal, 0) = mPressureScaling * tTau
        * (mPressureScaling * aPressureGrad(aCellOrdinal, 0) - aProjectedPressureGrad(aCellOrdinal, 0));

    aStabilization(aCellOrdinal, 1) = mPressureScaling * tTau
        * (mPressureScaling * aPressureGrad(aCellOrdinal, 1) - aProjectedPressureGrad(aCellOrdinal, 1));
}

template<>
template<typename ConfigT, typename PressGradT, typename ProjPressGradT, typename ResultT>
DEVICE_TYPE inline void
ComputeStabilization<1>::operator()(const Plato::OrdinalType & aCellOrdinal,
                                    const Plato::ScalarVectorT<ConfigT> & aCellVolume,
                                    const Plato::ScalarMultiVectorT<PressGradT> & aPressureGrad,
                                    const Plato::ScalarMultiVectorT<ProjPressGradT> & aProjectedPressureGrad,
                                    const Plato::ScalarMultiVectorT<ResultT> & aStabilization)
{
    ConfigT tTau = pow(aCellVolume(aCellOrdinal), mTwoOverThree) / (static_cast<Plato::Scalar>(2.0) * mElasticShearModulus);
    aStabilization(aCellOrdinal, 0) = mPressureScaling * tTau
        * (mPressureScaling * aPressureGrad(aCellOrdinal, 0) - aProjectedPressureGrad(aCellOrdinal, 0));
}








/**************************************************************************//**
 * \brief Evaluate stabilized elasto-plastic residual, defined as
 *
 * \f$   \langle \nabla{v_h}, s_h \rangle + \langle \nabla\cdot{v_h}, p_h \rangle
 *     - \langle v_h, f \rangle - \langle v_h, b \rangle = 0\ \forall\ \v_h \in V_{h,0}
 *       = \{v_h \in V_h | v = 0\ \mbox{in}\ \partial\Omega_{u} \} \f$
 *
 * \f$   \langle q_h, \nabla\cdot{u_h} \rangle - \langle q_h, \frac{1}{K}p_h \rangle
 *     - \sum_{e=1}^{N_{elem}} \tau_e \langle \nabla{q_h} \left[ \nabla{p_h} - \Pi_h
 *       \right] \rangle = 0\ \forall\ q_h \in L_h \subset\ L^2(\Omega) \f$
 *
 * \f$ \langle \nabla{p_h}, \eta_h \rangle - \langle \Pi_h, \eta_h \rangle = 0\
 *     \forall\ \eta_h \in V_h \subset\ H^{1}(\Omega) \f$
 *
 ******************************************************************************/
template<typename EvaluationType, typename PhysicsType>
class ElastoPlasticityResidual: public Plato::AbstractVectorFunctionVMSInc<EvaluationType>
{
// Private member data
private:
    static constexpr auto mSpaceDim = EvaluationType::SpatialDim;               /*!< spatial dimensions */
    static constexpr auto mNumVoigtTerms = PhysicsType::mNumVoigtTerms;         /*!< number of voigt terms */
    static constexpr auto mNumDofsPerCell = PhysicsType::mNumDofsPerCell;       /*!< number of degrees of freedom (dofs) per cell */
    static constexpr auto mNumDofsPerNode = PhysicsType::mNumDofsPerNode;       /*!< number of dofs per node */
    static constexpr auto mNumNodesPerCell = PhysicsType::mNumNodesPerCell;     /*!< number nodes per cell */
    static constexpr auto mPressureDofOffset = PhysicsType::mPressureDofOffset; /*!< number of pressure dofs offset */

    static constexpr auto mNumMechDims = mSpaceDim;         /*!< number of mechanical degrees of freedom */
    static constexpr Plato::OrdinalType mMechDofOffset = 0; /*!< mechanical degrees of freedom offset */

    using Plato::AbstractVectorFunctionVMSInc<EvaluationType>::mMesh;     /*!< mesh database */
    using Plato::AbstractVectorFunctionVMSInc<EvaluationType>::mDataMap;  /*!< PLATO Engine output database */
    using Plato::AbstractVectorFunctionVMSInc<EvaluationType>::mMeshSets; /*!< side-sets metadata */

    using GlobalStateT = typename EvaluationType::StateScalarType;             /*!< global state variables automatic differentiation type */
    using PrevGlobalStateT = typename EvaluationType::PrevStateScalarType;     /*!< global state variables automatic differentiation type */
    using LocalStateT = typename EvaluationType::LocalStateScalarType;         /*!< local state variables automatic differentiation type */
    using PrevLocalStateT = typename EvaluationType::PrevLocalStateScalarType; /*!< local state variables automatic differentiation type */
    using NodeStateT = typename EvaluationType::NodeStateScalarType;           /*!< node State AD type */
    using ControlT = typename EvaluationType::ControlScalarType;               /*!< control variables automatic differentiation type */
    using ConfigT = typename EvaluationType::ConfigScalarType;                 /*!< config variables automatic differentiation type */
    using ResultT = typename EvaluationType::ResultScalarType;                 /*!< result variables automatic differentiation type */

    Plato::Scalar mPoissonsRatio;                  /*!< Poisson's ratio */
    Plato::Scalar mElasticModulus;                 /*!< elastic modulus */
    Plato::Scalar mElasticBulkModulus;             /*!< elastic bulk modulus */
    Plato::Scalar mElasticShearModulus;            /*!< elastic shear modulus */
    Plato::Scalar mElasticPropertiesPenaltySIMP;   /*!< SIMP penalty for elastic properties */
    Plato::Scalar mElasticPropertiesMinErsatzSIMP; /*!< SIMP min ersatz stiffness for elastic properties */

    std::vector<std::string> mPlotTable; /*!< array with output data identifiers */

    std::shared_ptr<Plato::BodyLoads<EvaluationType>> mBodyLoads;                                                /*!< body loads interface */
    std::shared_ptr<Plato::LinearTetCubRuleDegreeOne<mSpaceDim>> mCubatureRule;                                  /*!< linear cubature rule */
    std::shared_ptr<Plato::NaturalBCs<mSpaceDim, mNumMechDims, mNumDofsPerNode, mMechDofOffset>> mBoundaryLoads; /*!< boundary loads interface */

// Private access functions
private:
    /******************************************************************************//**
     * \brief initialize material, loads and output data
     * \param [in] aProblemParams input XML data
     **********************************************************************************/
    void initialize(Teuchos::ParameterList &aProblemParams)
    {
        auto tMaterialParamList = aProblemParams.get<Teuchos::ParameterList>("Material Model");
        this->parseIsotropicElasticMaterialProperties(tMaterialParamList);
        this->parseExternalForces(aProblemParams);

        auto tResidualParams = aProblemParams.sublist("Elliptic");
        if (tResidualParams.isType < Teuchos::Array < std::string >> ("Plottable"))
        {
            mPlotTable = tResidualParams.get < Teuchos::Array < std::string >> ("Plottable").toVector();
        }
    }

    /**************************************************************************//**
    * \brief Parse external forces
    * \param [in] aProblemParams Teuchos parameter list
    ******************************************************************************/
    void parseExternalForces(Teuchos::ParameterList &aProblemParams)
    {
        // Parse body loads
        if (aProblemParams.isSublist("Body Loads"))
        {
            mBodyLoads = std::make_shared<Plato::BodyLoads<EvaluationType>>(aProblemParams.sublist("Body Loads"));
        }

        // Parse Neumann conditions
        if (aProblemParams.isSublist("Mechanical Natural Boundary Conditions"))
        {
            mBoundaryLoads = std::make_shared<Plato::NaturalBCs<mSpaceDim, mNumMechDims, mNumDofsPerNode, mMechDofOffset>>
                (aProblemParams.sublist("Mechanical Natural Boundary Conditions"));
        }
    }

    /**************************************************************************//**
    * \brief Parse isotropic material parameters
    * \param [in] aProblemParams Teuchos parameter list
    ******************************************************************************/
    void parseIsotropicElasticMaterialProperties(Teuchos::ParameterList &aMaterialParamList)
    {
        if (aMaterialParamList.isSublist("Isotropic Linear Elastic"))
        {
            auto tElasticSubList = aMaterialParamList.sublist("Isotropic Linear Elastic");
            mPoissonsRatio = Plato::parse_poissons_ratio(tElasticSubList);
            mElasticModulus = Plato::parse_elastic_modulus(tElasticSubList);
            mElasticBulkModulus = Plato::compute_bulk_modulus(mElasticModulus, tPoissonsRatio);
            mElasticShearModulus = Plato::compute_shear_modulus(mElasticModulus, tPoissonsRatio);
            this->parsePressureTermScaling(aMaterialParamList)
        }
        else
        {
            THROWERR("'Isotropic Linear Elastic' sublist of 'Material Model' is not define.")
        }
    }

    /**************************************************************************//**
    * \brief Parse pressure scaling, needed to minimize the linear system's condition number.
    * \param [in] aProblemParams Teuchos parameter list
    ******************************************************************************/
    void parsePressureTermScaling(Teuchos::ParameterList & aMaterialParamList)
    {
        if (paramList.isType<Plato::Scalar>("Pressure Scaling"))
        {
            mPressureScaling = aMaterialParamList.get<Plato::Scalar>("Pressure Scaling");
        }
        else
        {
            mPressureScaling = mElasticBulkModulus;
        }
    }

    /**************************************************************************//**
    * \brief Copy data to output data map
    * \tparam DataT data type
    * \param [in] aData output data
    * \param [in] aName output data name
    ******************************************************************************/
    template<typename DataT>
    void outputData(const DataT & aData, const std::string & aName)
    {
        if(std::count(mPlotTable.begin(), mPlotTable.end(), aName))
        {
            toMap(mDataMap, aData, aName);
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
    ElastoPlasticityResidual(Omega_h::Mesh &aMesh,
                             Omega_h::MeshSets &aMeshSets,
                             Plato::DataMap &aDataMap,
                             Teuchos::ParameterList &aProblemParams) :
        Plato::AbstractVectorFunctionVMSInc<EvaluationType>(aMesh, aMeshSets, aDataMap),
        mPoissonsRatio(-1.0),
        mElasticModulus(-1.0),
        mElasticBulkModulus(-1.0),
        mElasticShearModulus(-1.0),
        mElasticPropertiesPenaltySIMP(3),
        mElasticPropertiesMinErsatzSIMP(1e-9),
        mBodyLoads(nullptr),
        mBoundaryLoads(nullptr),
        mCubatureRule(std::make_shared<Plato::LinearTetCubRuleDegreeOne<mSpaceDim>>())
    {
        this->initialize(aProblemParams);
    }

    /******************************************************************************//**
     * \brief Destructor
     **********************************************************************************/
    virtual ~ElastoPlasticityResidual()
    {
    }

    void setIsotropicLinearElasticProperties(const Plato::Scalar & aElasticModulus, const Plato::Scalar & aPoissonsRatio)
    {
        mPoissonsRatio = aPoissonsRatio;
        mElasticModulus = aElasticModulus;
        mElasticBulkModulus = Plato::compute_bulk_modulus(mElasticModulus, mPoissonsRatio);
        mElasticShearModulus = Plato::compute_shear_modulus(mElasticModulus, mPoissonsRatio);
    }

    /****************************************************************************//**
     * \brief Add external forces to residual
     * \param [in] aGlobalState current global state ( i.e. state at the n-th time interval (\f$ t^{n} \f$) )
     * \param [in] aControl design variables
     * \param [in/out] aResult residual evaluation
     ********************************************************************************/
    void addExternalForces(const Plato::ScalarMultiVectorT<GlobalStateT> &aGlobalState,
                           const Plato::ScalarMultiVectorT<ControlT> &aControl,
                           const Plato::ScalarMultiVectorT<ResultT> &aResult)
    {
        if (mBodyLoads != nullptr)
        {
            Plato::Scalar tScale = -1.0;
            mBodyLoads->get(mMesh, aGlobalState, aControl, aResult, tScale);
        }

        if (mBoundaryLoads != nullptr)
        {
            Plato::Scalar tScale = -1.0;
            mBoundaryLoads->get(&mMesh, mMeshSets, aGlobalState, aControl, aResult, tScale);
        }
    }

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
    void evaluate(const Plato::ScalarMultiVectorT<GlobalStateT> &aGlobalState,
                  const Plato::ScalarMultiVectorT<PrevGlobalStateT> &aGlobalStatePrev,
                  const Plato::ScalarMultiVectorT<LocalStateT> &aLocalState,
                  const Plato::ScalarMultiVectorT<PrevLocalStateT> &aLocalStatePrev,
                  const Plato::ScalarMultiVectorT<NodeStateT> &aProjectedPressureGrad,
                  const Plato::ScalarMultiVectorT<ControlT> &aControl,
                  const Plato::ScalarArray3DT<ConfigT> &aConfig,
                  const Plato::ScalarMultiVectorT<ResultT> &aResult,
                  Plato::Scalar aTimeStep = 0.0)
    {
        auto tNumCells = mMesh.nelems();

        using GradScalarT = typename Plato::fad_type_t<PhysicsType, GlobalStateT, ConfigT>;
        using ElasticStrainT = typename Plato::fad_type_t<PhysicsType, LocalStateT, ConfigT, GlobalStateT>;

        // Functors used to compute residual-related quantities
        Plato::ScalarGrad<mSpaceDim> tComputeScalarGrad;
        Plato::ComputeGradientWorkset<mSpaceDim> tComputeGradient;
        Plato::DisplacementDivergence<mSpaceDim> tComputeDispDivergence;
        Plato::ThermoPlasticityUtilities<EvaluationType::SpatialDim, PhysicsType> tPlasticityUtils;
        Plato::ComputeStabilization<mSpaceDim> tComputeStabilization(mPressureScaling, mElasticShearModulus);
        Plato::InterpolateFromNodal<mSpaceDim, mNumDofsPerNode, mPressureDofOffset> tInterpolatePressureFromNodal;
        Plato::InterpolateFromNodal<mSpaceDim, mSpaceDim, 0 /* dof offset */, mSpaceDim> tInterpolatePressGradFromNodal;

        // Residual evaulation functors
        Plato::PressureDivergence<mSpaceDim, mNumDofsPerNode> tPressureDivergence;
        Plato::StressDivergence<mSpaceDim, mNumDofsPerNode, mMechDofOffset> tStressDivergence;
        Plato::ProjectToNode<mSpaceDim, mNumDofsPerNode, mPressureDofOffset> tProjectVolumeStrain;
        Plato::FluxDivergence<mSpaceDim, mNumDofsPerNode, mPressureDofOffset> tStabilizedDivergence;
        Plato::MSIMP tPenaltyFunction(mElasticPropertiesPenaltySIMP, mElasticPropertiesMinErsatzSIMP);

        Plato::ScalarVectorT<ResultT> tPressure("L2 pressure", tNumCells);
        Plato::ScalarVectorT<ConfigT> tCellVolume("cell volume", tNumCells);
        Plato::ScalarVectorT<ResultT> tVolumeStrain("volume strain", tNumCells);
        Plato::ScalarMultiVectorT<ResultT> tStabilization("cell stabilization", tNumCells, mSpaceDim);
        Plato::ScalarMultiVectorT<GradScalarT> tPressureGrad("pressure gradient", tNumCells, mSpaceDim);
        Plato::ScalarMultiVectorT<ResultT> tDeviatoricStress("deviatoric stress", tNumCells, mNumVoigtTerms);
        Plato::ScalarMultiVectorT<ElasticStrainT> tElasticStrain("elastic strain", tNumCells, mNumVoigtTerms);
        Plato::ScalarArray3DT<ConfigT> tConfigurationGradient("configuration gradient", tNumCells, mNumNodesPerCell, mSpaceDim);
        Plato::ScalarMultiVectorT<NodeStateT> tProjectedPressureGradGP("projected pressure gradient - gauss pt", tNumCells, mSpaceDim);

        // Transfer elasticity parameters to device
        auto tPressureScaling = mPressureScaling;
        auto tElasticBulkModulus = mElasticBulkModulus;
        auto tElasticShearModulus = mElasticShearModulus;

        auto tQuadratureWeight = mCubatureRule->getCubWeight();
        auto tBasisFunctions = mCubatureRule->getBasisFunctions();
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType &aCellOrdinal)
        {
            // compute configuration gradients
            tComputeGradient(aCellOrdinal, tConfigurationGradient, aConfig, tCellVolume);
            tCellVolume(aCellOrdinal) *= tQuadratureWeight;

            // compute elastic strain, i.e. e_elastic = e_total - e_plastic
            tPlasticityUtils.computeElasticStrain(aCellOrdinal, aGlobalState, aLocalState,
                                                  tBasisFunctions, tConfigurationGradient, tElasticStrain);

            // compute pressure gradient
            tComputeScalarGrad(aCellOrdinal, mNumDofsPerNode, mPressureDofOffset,
                               aGlobalState, tConfigurationGradient, tPressureGrad);

            // interpolate projected pressure grad, pressure, and temperature to gauss point
            tInterpolatePressureFromNodal(aCellOrdinal, tBasisFunctions, aGlobalState, tPressure);
            tInterpolatePressGradFromNodal(aCellOrdinal, tBasisFunctions, aProjectedPressureGrad, tProjectedPressureGradGP);

            // compute cell penalty
            ControlT tDensity = Plato::cell_density<mNumNodesPerCell>(aCellOrdinal, aControl);
            ControlT tElasticPropertiesPenalty = tPenaltyFunction(tDensity);

            // compute deviatoric stress and displacement divergence
            ControlT tPenalizedShearModulus = tElasticPropertiesPenalty * tElasticShearModulus;
            tPlasticityUtils.computeDeviatoricStress(aCellOrdinal, tElasticStrain, tPenalizedShearModulus, tDeviatoricStress);
            ResultT tDispDivergence = tComputeDispDivergence(aCellOrdinal, tElasticStrain);

            // compute volume difference
            tVolumeStrain(aCellOrdinal) = tPressureScaling * tElasticPropertiesPenalty
                * (tDispDivergence - tPressure(aCellOrdinal) / tElasticBulkModulus);
            tPressure(aCellOrdinal) *= tPressureScaling * tElasticPropertiesPenalty;

            // compute cell stabilization term
            tComputeStabilization(aCellOrdinal, tCellVolume, tPressureGrad, tProjectedPressureGradGP, tStabilization);
            Plato::apply_penalty<mSpaceDim>(aCellOrdinal, tElasticPropertiesPenalty, tStabilization);

            // compute residual
            tStressDivergence (aCellOrdinal, aResult, tDeviatoricStress, tConfigurationGradient, tCellVolume);
            tPressureDivergence (aCellOrdinal, aResult, tPressure, tConfigurationGradient, tCellVolume);
            tStabilizedDivergence(aCellOrdinal, aResult, tStabilization, tConfigurationGradient, tCellVolume, -1.0);
            tProjectVolumeStrain (aCellOrdinal, tCellVolume, tBasisFunctions, tVolumeStrain, aResult);
        }, "stabilized elasto-plastic residual");

        this->addExternalForces(aGlobalState, aControl, aResult);
        this->outputData(tDeviatoricStress, "deviatoric stress");
        this->outputData(tPressure, "pressure");
    }
};
// class StabilizedElastoPlasticResidual







namespace ElastoPlasticityFactory
{

/******************************************************************************/
struct FunctionFactory
{
    /******************************************************************************//**
     * \brief Create a PLATO local vector function  inc (i.e. local residual equations)
     * \param [in] aMesh mesh database
     * \param [in] aMeshSets side sets database
     * \param [in] aDataMap output data database
     * \param [in] aInputParams input parameters
     * \param [in] aFunctionName vector function name
     * \return shared pointer to a stabilized vector function integrated in time
    **********************************************************************************/
    template<typename EvaluationType>
    std::shared_ptr<Plato::AbstractVectorFunctionVMSInc<EvaluationType>>
    createVectorFunctionVMSInc(Omega_h::Mesh& aMesh, Omega_h::MeshSets& aMeshSets, Plato::DataMap& aDataMap, Teuchos::ParameterList& aInputParams, std::string aFunctionName)
    {
        if(aFunctionName == "ElastoPlasticity")
        {
            constexpr auto tSpaceDim = EvaluationType::SpatialDim;
            return std::make_shared<Plato::ElastoPlasticityResidual<EvaluationType, Plato::SimplexElastoPasticity<tSpaceDim>> > (aMesh, aMeshSets, aDataMap, aInputParams);
        }
        else
        {
            const std::string tError = std::string("Unknown createVectorFunctionVMSInc '") + aFunctionName + "' specified.";
            THROWERR(tError)
        }
    }
};
// struct FunctionFactory

}
// namespace ElastoPlasticityFactory

/****************************************************************************//**
 * \brief Concrete class defining the Physics Type template argument for a
 * VectorFunctionVMSInc.  A VectorFunctionVMSInc is defined by a stabilized
 * Partial Differential Equation (PDE) integrated implicitly in time.  The
 * stabilization technique is based on the Variational Multiscale (VMS) method.
 * Here, the (Inc) in VectorFunctionVMSInc denotes increment.
 *******************************************************************************/
template<Plato::OrdinalType NumSpaceDim>
class ElastoPlasticity: public Plato::SimplexElastoPasticity<NumSpaceDim>
{
public:
    static constexpr auto SpaceDim = NumSpaceDim;
    using SimplexT = Plato::SimplexPlasticity<NumSpaceDim>;
    typedef Plato::ElastoPlasticityFactory::FunctionFactory FunctionFactory;
};
// class ElastoPlasticity






template<typename PhysicsT>
class VectorFunctionVMSInc
{
// Private access member data
private:
    using Residual        = typename Plato::Evaluation<PhysicsT>::Residual;       /*!< automatic differentiation (AD) type for the residual */
    using GradientX       = typename Plato::Evaluation<PhysicsT>::GradientX;      /*!< AD type for the configuration */
    using GradientZ       = typename Plato::Evaluation<PhysicsT>::GradientZ;      /*!< AD type for the controls */
    using JacobianPgrad   = typename Plato::Evaluation<PhysicsT>::JacobianN;      /*!< AD type for the nodal pressure gradient */
    using LocalJacobian   = typename Plato::Evaluation<PhysicsT>::LocalJacobian;  /*!< AD type for the current local states */
    using LocalJacobianP  = typename Plato::Evaluation<PhysicsT>::LocalJacobianP; /*!< AD type for the previous local states */
    using GlobalJacobian  = typename Plato::Evaluation<PhysicsT>::Jacobian;       /*!< AD type for the current global states */
    using GlobalJacobianP = typename Plato::Evaluation<PhysicsT>::JacobianP;      /*!< AD type for the previous global states */

    static constexpr auto mNumControl = PhysicsT::mNumControl;                        /*!< number of control fields, i.e. vectors, number of materials */
    static constexpr auto mNumSpatialDims = PhysicsT::mNumSpatialDims;                /*!< number of spatial dimensions */
    static constexpr auto mNumNodesPerCell = PhysicsT::mNumNodesPerCell;              /*!< number of nodes per cell (i.e. element) */
    static constexpr auto mNumGlobalDofsPerNode = PhysicsT::mNumDofsPerNode;          /*!< number of global degrees of freedom per node */
    static constexpr auto mNumGlobalDofsPerCell = PhysicsT::mNumDofsPerCell;          /*!< number of global degrees of freedom per cell (i.e. element) */
    static constexpr auto mNumLocalDofsPerCell = PhysicsT::mNumLocalDofsPerCell;      /*!< number of local degrees of freedom per cell (i.e. element) */
    static constexpr auto mNumNodeStatePerNode = PhysicsT::mNumNodeStatePerNode;      /*!< number of pressure gradient degrees of freedom per node */
    static constexpr auto mNumNodeStatePerCell = PhysicsT::mNumNodeStatePerCell;      /*!< number of pressure gradient degrees of freedom per cell (i.e. element) */
    static constexpr auto mNumConfigDofsPerCell = mNumSpatialDims * mNumNodesPerCell; /*!< number of configuration (i.e. coordinates) degrees of freedom per cell (i.e. element) */

    const Plato::OrdinalType mNumNodes; /*!< total number of nodes */
    const Plato::OrdinalType mNumCells; /*!< total number of cells (i.e. elements) */

    Plato::DataMap& mDataMap;  /*!< output data map */
    Plato::WorksetBase<PhysicsT> mWorksetBase; /*!< assembly routine interface */

    std::shared_ptr<Plato::AbstractVectorFunctionVMSInc<Residual>>        mGlobalVecFuncResidual;   /*!< global vector function residual */
    std::shared_ptr<Plato::AbstractVectorFunctionVMSInc<GradientX>>       mGlobalVecFuncJacobianX;  /*!< global vector function Jacobian wrt configuration */
    std::shared_ptr<Plato::AbstractVectorFunctionVMSInc<GradientZ>>       mGlobalVecFuncJacobianZ;  /*!< global vector function Jacobian wrt controls */
    std::shared_ptr<Plato::AbstractVectorFunctionVMSInc<JacobianPgrad>>   mGlobalVecFuncJacPgrad;   /*!< global vector function Jacobian wrt projected pressure gradient */
    std::shared_ptr<Plato::AbstractVectorFunctionVMSInc<LocalJacobian>>   mGlobalVecFuncJacobianC;  /*!< global vector function Jacobian wrt current local states */
    std::shared_ptr<Plato::AbstractVectorFunctionVMSInc<LocalJacobianP>>  mGlobalVecFuncJacobianCP; /*!< global vector function Jacobian wrt previous local states */
    std::shared_ptr<Plato::AbstractVectorFunctionVMSInc<GlobalJacobian>>  mGlobalVecFuncJacobianU;  /*!< global vector function Jacobian wrt current global states */
    std::shared_ptr<Plato::AbstractVectorFunctionVMSInc<GlobalJacobianP>> mGlobalVecFuncJacobianUP; /*!< global vector function Jacobian wrt previous global states */

// Public access functions
public:
    /**************************************************************************//**
     * \brief Constructor
     * \param [in] aMesh mesh data base
     * \param [in] aMeshSets mesh sets data base
     * \param [in] aDataMap problem-specific data map
     * \param [in] aParamList Teuchos parameter list with input data
     * \param [in] aVectorFuncType vector function type string
     ******************************************************************************/
    VectorFunctionVMSInc(Omega_h::Mesh& aMesh,
                         Omega_h::MeshSets& aMeshSets,
                         Plato::DataMap& aDataMap,
                         Teuchos::ParameterList& aParamList,
                         std::string& aVectorFuncType) :
            mNumNodes(aMesh.nverts()),
            mNumCells(aMesh.nelems()),
            mDataMap(aDataMap),
            mWorksetBase(aMesh)
    {
        typename PhysicsT::FunctionFactory tFunctionFactory;

        mGlobalVecFuncResidual = tFunctionFactory.template createVectorFunctionVMSInc<Residual>
                                                           (aMesh, aMeshSets, aDataMap, aParamList, aVectorFuncType);

        mGlobalVecFuncJacobianU = tFunctionFactory.template createVectorFunctionVMSInc<GlobalJacobian>
                                                            (aMesh, aMeshSets, aDataMap, aParamList, aVectorFuncType);

        mGlobalVecFuncJacobianUP = tFunctionFactory.template createVectorFunctionVMSInc<GlobalJacobianP>
                                                             (aMesh, aMeshSets, aDataMap, aParamList, aVectorFuncType);

        mGlobalVecFuncJacobianC = tFunctionFactory.template createVectorFunctionVMSInc<LocalJacobian>
                                                            (aMesh, aMeshSets, aDataMap, aParamList, aVectorFuncType);

        mGlobalVecFuncJacobianCP = tFunctionFactory.template createVectorFunctionVMSInc<LocalJacobianP>
                                                             (aMesh, aMeshSets, aDataMap, aParamList, aVectorFuncType);

        mGlobalVecFuncJacobianZ = tFunctionFactory.template createVectorFunctionVMSInc<GradientZ>
                                                            (aMesh, aMeshSets, aDataMap, aParamList, aVectorFuncType);

        mGlobalVecFuncJacobianX = tFunctionFactory.template createVectorFunctionVMSInc<GradientX>
                                                            (aMesh, aMeshSets, aDataMap, aParamList, aVectorFuncType);

        mGlobalVecFuncJacPgrad = tFunctionFactory.template createVectorFunctionVMSInc<JacobianPgrad>
                                                           (aMesh, aMeshSets, aDataMap, aParamList, aVectorFuncType);
    }

    /**************************************************************************//**
     * \brief Destructor
    ******************************************************************************/
    ~VectorFunctionVMSInc(){}

    /**************************************************************************//**
    * \brief Compute the global residual vector
    * \param [in] aGlobalState global state at current time step
    * \param [in] aPrevGlobalState global state at previous time step
    * \param [in] aLocalState local state at current time step
    * \param [in] aPrevLocalState local state at previous time step
    * \param [in] aControl control parameters
    * \param [in] aNodeState pressure gradient
    * \param [in] aTimeStep time step
    * \return Global residual vector
    ******************************************************************************/
    Plato::ScalarVectorT<typename Residual::ResultScalarType>
    value(const Plato::ScalarVector & aGlobalState,
          const Plato::ScalarVector & aPrevGlobalState,
          const Plato::ScalarVector & aLocalState,
          const Plato::ScalarVector & aPrevLocalState,
          const Plato::ScalarVector & aNodeState,
          const Plato::ScalarVector & aControl,
          Plato::Scalar aTimeStep = 0.0) const
    {
        // Workset config
        using ConfigScalar = typename Residual::ConfigScalarType;
        Plato::ScalarArray3DT<ConfigScalar>
            tConfigWS("Config Workset", mNumCells, mNumNodesPerCell, mNumSpatialDims);
        mWorksetBase.worksetConfig(tConfigWS);

        // Workset current global state
        using GlobalStateScalar = typename Residual::StateScalarType;
        Plato::ScalarMultiVectorT<GlobalStateScalar>
            tGlobalStateWS("Current Global State Workset", mNumCells, mNumGlobalDofsPerCell);
        mWorksetBase.worksetState(aGlobalState, tGlobalStateWS);

        // Workset previous global state
        using PrevGlobalStateScalar = typename Residual::PrevStateScalarType;
        Plato::ScalarMultiVectorT<PrevGlobalStateScalar>
            tPrevGlobalStateWS("Previous Global State Workset", mNumCells, mNumGlobalDofsPerCell);
        mWorksetBase.worksetState(aPrevGlobalState, tPrevGlobalStateWS);

        // Workset local state
        using LocalStateScalar = typename Residual::LocalStateScalarType;
        Plato::ScalarMultiVectorT<LocalStateScalar>
            tLocalStateWS("Local Current State Workset", mNumCells, mNumLocalDofsPerCell);
        mWorksetBase.worksetLocalState(aLocalState, tLocalStateWS);

        // Workset previous local state
        using PrevLocalStateScalar = typename Residual::PrevLocalStateScalarType;
        Plato::ScalarMultiVectorT<PrevLocalStateScalar>
            tPrevLocalStateWS("Previous Local State Workset", mNumCells, mNumLocalDofsPerCell);
        mWorksetBase.worksetLocalState(aPrevLocalState, tPrevLocalStateWS);

        // Workset node state, i.e. projected pressure gradient
        using NodeStateScalar = typename Residual::NodeStateScalarType;
        Plato::ScalarMultiVectorT<NodeStateScalar> tNodeStateWS("Node State Workset", mNumCells, mNumNodeStatePerCell);
        mWorksetBase.worksetNodeState(aNodeState, tNodeStateWS);

        // Workset control
        using ControlScalar = typename Residual::ControlScalarType;
        Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", mNumCells, mNumNodesPerCell);
        mWorksetBase.worksetControl(aControl, tControlWS);

        // Workset residual
        using ResultScalar = typename Residual::ResultScalarType;
        Plato::ScalarMultiVectorT<ResultScalar> tResidualWS("Residual Workset", mNumCells, mNumGlobalDofsPerCell);

        // Evaluate global residual
        mGlobalVecFuncResidual->evaluate(tGlobalStateWS, tPrevGlobalStateWS, tLocalStateWS, tPrevLocalStateWS,
                                         tNodeStateWS, tControlWS, tConfigWS, tResidualWS, aTimeStep);

        // create and assemble to return view
        const auto tTotalNumDofs = mNumGlobalDofsPerNode * mNumNodes;
        Kokkos::View<Plato::Scalar*, Kokkos::LayoutRight, Plato::MemSpace>  tAssembledResidual("Assembled Residual", tTotalNumDofs);
        mWorksetBase.assembleResidual( tResidualWS, tAssembledResidual );

        return tAssembledResidual;
    }

    /**************************************************************************//**
    * \brief Compute Jacobian with respect to (wrt) control of the global residual
    * \param [in] aGlobalState global state at current time step
    * \param [in] aPrevGlobalState global state at previous time step
    * \param [in] aLocalState local state at current time step
    * \param [in] aPrevLocalState local state at previous time step
    * \param [in] aControl control parameters
    * \param [in] aNodeState pressure gradient
    * \param [in] aTimeStep time step
    * \return Jacobian wrt control of the global residual
    ******************************************************************************/
    Teuchos::RCP<Plato::CrsMatrixType>
    gradient_z(const Plato::ScalarVector & aGlobalState,
               const Plato::ScalarVector & aPrevGlobalState,
               const Plato::ScalarVector & aLocalState,
               const Plato::ScalarVector & aPrevLocalState,
               const Plato::ScalarVector & aNodeState,
               const Plato::ScalarVector & aControl,
               Plato::Scalar aTimeStep = 0.0) const
    {
        // Workset config
        using ConfigScalar = typename GradientZ::ConfigScalarType;
        Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset", mNumCells, mNumNodesPerCell, mNumSpatialDims);
        mWorksetBase.worksetConfig(tConfigWS);

        // Workset current global state
        using GlobalStateScalar = typename GradientZ::StateScalarType;
        Plato::ScalarMultiVectorT<GlobalStateScalar> tGlobalStateWS("Current Global State Workset", mNumCells, mNumGlobalDofsPerCell);
        mWorksetBase.worksetState(aGlobalState, tGlobalStateWS);

        // Workset previous global state
        using PrevGlobalStateScalar = typename GradientZ::PrevStateScalarType;
        Plato::ScalarMultiVectorT<PrevGlobalStateScalar> tPrevGlobalStateWS("Previous Global State Workset", mNumCells, mNumGlobalDofsPerCell);
        mWorksetBase.worksetState(aPrevGlobalState, tPrevGlobalStateWS);

        // Workset local state
        using LocalStateScalar = typename GradientZ::LocalStateScalarType;
        Plato::ScalarMultiVectorT<LocalStateScalar> tLocalStateWS("Local Current State Workset", mNumCells, mNumLocalDofsPerCell);
        mWorksetBase.worksetLocalState(aLocalState, tLocalStateWS);

        // Workset previous local state
        using PrevLocalStateScalar = typename GradientZ::PrevLocalStateScalarType;
        Plato::ScalarMultiVectorT<PrevLocalStateScalar> tPrevLocalStateWS("Previous Local State Workset", mNumCells, mNumLocalDofsPerCell);
        mWorksetBase.worksetLocalState(aPrevLocalState, tPrevLocalStateWS);

        // Workset node state, i.e. projected pressure gradient
        using NodeStateScalar = typename GradientZ::NodeStateScalarType;
        Plato::ScalarMultiVectorT<NodeStateScalar> tNodeStateWS("Node State Workset", mNumCells, mNumNodeStatePerCell);
        mWorksetBase.worksetNodeState(aNodeState, tNodeStateWS);

        // Workset control
        using ControlScalar = typename GradientZ::ControlScalarType;
        Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", mNumCells, mNumNodesPerCell);
        mWorksetBase.worksetControl(aControl, tControlWS);

        // Create Jacobian workset
        using JacobianScalar = typename GradientZ::ResultScalarType;
        Plato::ScalarMultiVectorT<JacobianScalar> tJacobianWS("Jacobian Control Workset", mNumCells, mNumGlobalDofsPerCell);

        // Call evaluate function - compute Jacobian wrt controls
        mGlobalVecFuncJacobianZ->evaluate(tGlobalStateWS, tPrevGlobalStateWS, tLocalStateWS, tPrevLocalStateWS,
                                          tNodeStateWS, tControlWS, tConfigWS, tJacobianWS, aTimeStep);

        // Create return Jacobain
        auto tMesh = mGlobalVecFuncJacobianZ->getMesh();
        Teuchos::RCP<Plato::CrsMatrixType> tAssembledJacobian =
                Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumControl, mNumGlobalDofsPerNode>(&tMesh);

        // Assemble Jacobian
        Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumControl, mNumGlobalDofsPerNode>
                tJacobianMatEntryOrdinal(tAssembledJacobian, &tMesh);
        auto tJacobianMatEntries = tAssembledJacobian->entries();
        mWorksetBase.assembleTransposeJacobian(mNumGlobalDofsPerCell, mNumNodesPerCell,
                                               tJacobianMatEntryOrdinal, tJacobianWS, tJacobianMatEntries);

        return tAssembledJacobian;
    }

    /**************************************************************************//**
    * \brief Compute Jacobian wrt configuration of the global residual
    * \param [in] aGlobalState global state at current time step
    * \param [in] aPrevGlobalState global state at previous time step
    * \param [in] aLocalState local state at current time step
    * \param [in] aPrevLocalState local state at previous time step
    * \param [in] aControl control parameters
    * \param [in] aNodeState pressure gradient
    * \param [in] aTimeStep time step
    * \return Jacobian wrt configuration of the global residual
    ******************************************************************************/
    Teuchos::RCP<Plato::CrsMatrixType>
    gradient_x(const Plato::ScalarVector & aGlobalState,
               const Plato::ScalarVector & aPrevGlobalState,
               const Plato::ScalarVector & aLocalState,
               const Plato::ScalarVector & aPrevLocalState,
               const Plato::ScalarVector & aNodeState,
               const Plato::ScalarVector & aControl,
               Plato::Scalar aTimeStep = 0.0) const
    {
        // Workset config
        using ConfigScalar = typename GradientX::ConfigScalarType;
        Plato::ScalarArray3DT<ConfigScalar>
            tConfigWS("Config Workset", mNumCells, mNumNodesPerCell, mNumSpatialDims);
        mWorksetBase.worksetConfig(tConfigWS);

        // Workset current global state
        using GlobalStateScalar = typename GradientX::StateScalarType;
        Plato::ScalarMultiVectorT<GlobalStateScalar>
            tGlobalStateWS("Current Global State Workset", mNumCells, mNumGlobalDofsPerCell);
        mWorksetBase.worksetState(aGlobalState, tGlobalStateWS);

        // Workset previous global state
        using PrevGlobalStateScalar = typename GradientX::PrevStateScalarType;
        Plato::ScalarMultiVectorT<PrevGlobalStateScalar>
            tPrevGlobalStateWS("Previous Global State Workset", mNumCells, mNumGlobalDofsPerCell);
        mWorksetBase.worksetState(aPrevGlobalState, tPrevGlobalStateWS);

        // Workset local state
        using LocalStateScalar = typename GradientX::LocalStateScalarType;
        Plato::ScalarMultiVectorT<LocalStateScalar>
            tLocalStateWS("Local Current State Workset", mNumCells, mNumLocalDofsPerCell);
        mWorksetBase.worksetLocalState(aLocalState, tLocalStateWS);

        // Workset previous local state
        using PrevLocalStateScalar = typename GradientX::PrevLocalStateScalarType;
        Plato::ScalarMultiVectorT<PrevLocalStateScalar>
            tPrevLocalStateWS("Previous Local State Workset", mNumCells, mNumLocalDofsPerCell);
        mWorksetBase.worksetLocalState(aPrevLocalState, tPrevLocalStateWS);

        // Workset node state, i.e. projected pressure gradient
        using NodeStateScalar = typename GradientX::NodeStateScalarType;
        Plato::ScalarMultiVectorT<NodeStateScalar> tNodeStateWS("Node State Workset", mNumCells, mNumNodeStatePerCell);
        mWorksetBase.worksetNodeState(aNodeState, tNodeStateWS);

        // Workset control
        using ControlScalar = typename GradientX::ControlScalarType;
        Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", mNumCells, mNumNodesPerCell);
        mWorksetBase.worksetControl(aControl, tControlWS);

        // create return view
        using JacobianScalar = typename GradientX::ResultScalarType;
        Plato::ScalarMultiVectorT<JacobianScalar> tJacobianWS("Jacobian Configuration", mNumCells, mNumGlobalDofsPerCell);

        // Call evaluate function - compute Jacobian wrt configuration
        mGlobalVecFuncJacobianX->evaluate(tGlobalStateWS, tPrevGlobalStateWS, tLocalStateWS, tPrevLocalStateWS,
                                          tNodeStateWS, tControlWS, tConfigWS, tJacobianWS, aTimeStep);

        // create return matrix
        auto tMesh = mGlobalVecFuncJacobianX->getMesh();
        Teuchos::RCP<Plato::CrsMatrixType> tAssembledJacobian =
                Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumSpatialDims, mNumGlobalDofsPerNode>(&tMesh);

        // Assemble Jacobian
        Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumSpatialDims, mNumGlobalDofsPerNode>
                tJacobianEntryOrdinal(tAssembledJacobian, &tMesh);

        auto tJacobianMatEntries = tAssembledJacobian->entries();
        mWorksetBase.assembleTransposeJacobian(mNumGlobalDofsPerCell, mNumConfigDofsPerCell,
                                               tJacobianEntryOrdinal, tJacobianWS, tJacobianMatEntries);

        return tAssembledJacobian;
    }

    /**************************************************************************//**
    * \brief Compute Jacobian wrt current global states of the global residual
    * \param [in] aGlobalState global state at current time step
    * \param [in] aPrevGlobalState global state at previous time step
    * \param [in] aLocalState local state at current time step
    * \param [in] aPrevLocalState local state at previous time step
    * \param [in] aControl control parameters
    * \param [in] aNodeState pressure gradient
    * \param [in] aTimeStep time step
    * \return Jacobian wrt current global states of the global residual
    ******************************************************************************/
    Plato::ScalarMultiVectorT<typename GlobalJacobian::ResultScalarType>
    gradient_u(const Plato::ScalarVector & aGlobalState,
               const Plato::ScalarVector & aPrevGlobalState,
               const Plato::ScalarVector & aLocalState,
               const Plato::ScalarVector & aPrevLocalState,
               const Plato::ScalarVector & aNodeState,
               const Plato::ScalarVector & aControl,
               Plato::Scalar aTimeStep = 0.0) const
    {
        // Workset config
        using ConfigScalar = typename GlobalJacobian::ConfigScalarType;
        Plato::ScalarArray3DT<ConfigScalar>
            tConfigWS("Config Workset", mNumCells, mNumNodesPerCell, mNumSpatialDims);
        mWorksetBase.worksetConfig(tConfigWS);

        // Workset current global state
        using GlobalStateScalar = typename GlobalJacobian::StateScalarType;
        Plato::ScalarMultiVectorT<GlobalStateScalar>
            tGlobalStateWS("Current Global State Workset", mNumCells, mNumGlobalDofsPerCell);
        mWorksetBase.worksetState(aGlobalState, tGlobalStateWS);

        // Workset previous global state
        using PrevGlobalStateScalar = typename GlobalJacobian::PrevStateScalarType;
        Plato::ScalarMultiVectorT<PrevGlobalStateScalar>
            tPrevGlobalStateWS("Previous Global State Workset", mNumCells, mNumGlobalDofsPerCell);
        mWorksetBase.worksetState(aPrevGlobalState, tPrevGlobalStateWS);

        // Workset local state
        using LocalStateScalar = typename GlobalJacobian::LocalStateScalarType;
        Plato::ScalarMultiVectorT<LocalStateScalar>
            tLocalStateWS("Local Current State Workset", mNumCells, mNumLocalDofsPerCell);
        mWorksetBase.worksetLocalState(aLocalState, tLocalStateWS);

        // Workset previous local state
        using PrevLocalStateScalar = typename GlobalJacobian::PrevLocalStateScalarType;
        Plato::ScalarMultiVectorT<PrevLocalStateScalar>
            tPrevLocalStateWS("Previous Local State Workset", mNumCells, mNumLocalDofsPerCell);
        mWorksetBase.worksetLocalState(aPrevLocalState, tPrevLocalStateWS);

        // Workset node state, i.e. projected pressure gradient
        using NodeStateScalar = typename GlobalJacobian::NodeStateScalarType;
        Plato::ScalarMultiVectorT<NodeStateScalar> tNodeStateWS("Node State Workset", mNumCells, mNumNodeStatePerCell);
        mWorksetBase.worksetNodeState(aNodeState, tNodeStateWS);

        // Workset control
        using ControlScalar = typename GlobalJacobian::ControlScalarType;
        Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", mNumCells, mNumNodesPerCell);
        mWorksetBase.worksetControl(aControl, tControlWS);

        // Workset Jacobian wrt current global states
        using JacobianScalar = typename GlobalJacobian::ResultScalarType;
        Plato::ScalarMultiVectorT<JacobianScalar> tJacobianWS("Jacobian Current State", mNumCells, mNumGlobalDofsPerCell);

        // Call evaluate function - compute Jacobian wrt the current global states
        mGlobalVecFuncJacobianU->evaluate(tGlobalStateWS, tPrevGlobalStateWS, tLocalStateWS, tPrevLocalStateWS,
                                          tNodeStateWS, tControlWS, tConfigWS, tJacobianWS, aTimeStep);

        return tJacobianWS;
    }

    /**************************************************************************//**
    * \brief Compute Jacobian wrt previous global states of the global residual
    * \param [in] aGlobalState global state at current time step
    * \param [in] aPrevGlobalState global state at previous time step
    * \param [in] aLocalState local state at current time step
    * \param [in] aPrevLocalState local state at previous time step
    * \param [in] aControl control parameters
    * \param [in] aNodeState pressure gradient
    * \param [in] aTimeStep time step
    * \return Jacobian wrt previous global states of the global residual
    ******************************************************************************/
    Plato::ScalarMultiVectorT<typename GlobalJacobianP::ResultScalarType>
    gradient_up(const Plato::ScalarVector & aGlobalState,
                const Plato::ScalarVector & aPrevGlobalState,
                const Plato::ScalarVector & aLocalState,
                const Plato::ScalarVector & aPrevLocalState,
                const Plato::ScalarVector & aNodeState,
                const Plato::ScalarVector & aControl,
                Plato::Scalar aTimeStep = 0.0) const
    {
        // Workset config
        using ConfigScalar = typename GlobalJacobianP::ConfigScalarType;
        Plato::ScalarArray3DT<ConfigScalar>
            tConfigWS("Config Workset", mNumCells, mNumNodesPerCell, mNumSpatialDims);
        mWorksetBase.worksetConfig(tConfigWS);

        // Workset current global state
        using GlobalStateScalar = typename GlobalJacobianP::StateScalarType;
        Plato::ScalarMultiVectorT<GlobalStateScalar>
            tGlobalStateWS("Current Global State Workset", mNumCells, mNumGlobalDofsPerCell);
        mWorksetBase.worksetState(aGlobalState, tGlobalStateWS);

        // Workset previous global state
        using PrevGlobalStateScalar = typename GlobalJacobianP::PrevStateScalarType;
        Plato::ScalarMultiVectorT<PrevGlobalStateScalar>
            tPrevGlobalStateWS("Previous Global State Workset", mNumCells, mNumGlobalDofsPerCell);
        mWorksetBase.worksetState(aPrevGlobalState, tPrevGlobalStateWS);

        // Workset local state
        using LocalStateScalar = typename GlobalJacobianP::LocalStateScalarType;
        Plato::ScalarMultiVectorT<LocalStateScalar>
            tLocalStateWS("Local Current State Workset", mNumCells, mNumLocalDofsPerCell);
        mWorksetBase.worksetLocalState(aLocalState, tLocalStateWS);

        // Workset previous local state
        using PrevLocalStateScalar = typename GlobalJacobianP::PrevLocalStateScalarType;
        Plato::ScalarMultiVectorT<PrevLocalStateScalar>
            tPrevLocalStateWS("Previous Local State Workset", mNumCells, mNumLocalDofsPerCell);
        mWorksetBase.worksetLocalState(aPrevLocalState, tPrevLocalStateWS);

        // Workset node state, i.e. projected pressure gradient
        using NodeStateScalar = typename GlobalJacobianP::NodeStateScalarType;
        Plato::ScalarMultiVectorT<NodeStateScalar> tNodeStateWS("Node State Workset", mNumCells, mNumNodeStatePerCell);
        mWorksetBase.worksetNodeState(aNodeState, tNodeStateWS);

        // Workset control
        using ControlScalar = typename GlobalJacobianP::ControlScalarType;
        Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", mNumCells, mNumNodesPerCell);
        mWorksetBase.worksetControl(aControl, tControlWS);

        // Workset Jacobian wrt current global states
        using JacobianScalar = typename GlobalJacobianP::ResultScalarType;
        Plato::ScalarMultiVectorT<JacobianScalar> tJacobianWS("Jacobian Previous State", mNumCells, mNumGlobalDofsPerCell);

        // Call evaluate function - compute Jacobian wrt the previous global states
        mGlobalVecFuncJacobianU->evaluate(tGlobalStateWS, tPrevGlobalStateWS, tLocalStateWS, tPrevLocalStateWS,
                                          tNodeStateWS, tControlWS, tConfigWS, tJacobianWS, aTimeStep);

        return tJacobianWS;
    }

    /**************************************************************************//**
    * \brief Compute Jacobian wrt current local state of the global residual
    * \param [in] aGlobalState global state at current time step
    * \param [in] aPrevGlobalState global state at previous time step
    * \param [in] aLocalState local state at current time step
    * \param [in] aPrevLocalState local state at previous time step
    * \param [in] aControl control parameters
    * \param [in] aNodeState pressure gradient
    * \param [in] aTimeStep time step
    * \return Jacobian wrt current local state of the global residual
    ******************************************************************************/
    Plato::ScalarMultiVectorT<typename LocalJacobian::ResultScalarType>
    gradient_c(const Plato::ScalarVector & aGlobalState,
               const Plato::ScalarVector & aPrevGlobalState,
               const Plato::ScalarVector & aLocalState,
               const Plato::ScalarVector & aPrevLocalState,
               const Plato::ScalarVector & aNodeState,
               const Plato::ScalarVector & aControl,
               Plato::Scalar aTimeStep = 0.0) const
    {
        // Workset config
        using ConfigScalar = typename LocalJacobian::ConfigScalarType;
        Plato::ScalarArray3DT<ConfigScalar>
            tConfigWS("Config Workset", mNumCells, mNumNodesPerCell, mNumSpatialDims);
        mWorksetBase.worksetConfig(tConfigWS);

        // Workset current global state
        using GlobalStateScalar = typename LocalJacobian::StateScalarType;
        Plato::ScalarMultiVectorT<GlobalStateScalar>
            tGlobalStateWS("Current Global State Workset", mNumCells, mNumGlobalDofsPerCell);
        mWorksetBase.worksetState(aGlobalState, tGlobalStateWS);

        // Workset previous global state
        using PrevGlobalStateScalar = typename LocalJacobian::PrevStateScalarType;
        Plato::ScalarMultiVectorT<PrevGlobalStateScalar>
            tPrevGlobalStateWS("Previous Global State Workset", mNumCells, mNumGlobalDofsPerCell);
        mWorksetBase.worksetState(aPrevGlobalState, tPrevGlobalStateWS);

        // Workset local state
        using LocalStateScalar = typename LocalJacobian::LocalStateScalarType;
        Plato::ScalarMultiVectorT<LocalStateScalar>
            tLocalStateWS("Local Current State Workset", mNumCells, mNumLocalDofsPerCell);
        mWorksetBase.worksetLocalState(aLocalState, tLocalStateWS);

        // Workset previous local state
        using PrevLocalStateScalar = typename LocalJacobian::PrevLocalStateScalarType;
        Plato::ScalarMultiVectorT<PrevLocalStateScalar>
            tPrevLocalStateWS("Previous Local State Workset", mNumCells, mNumLocalDofsPerCell);
        mWorksetBase.worksetLocalState(aPrevLocalState, tPrevLocalStateWS);

        // Workset node state, i.e. projected pressure gradient
        using NodeStateScalar = typename LocalJacobian::NodeStateScalarType;
        Plato::ScalarMultiVectorT<NodeStateScalar> tNodeStateWS("Node State Workset", mNumCells, mNumNodeStatePerCell);
        mWorksetBase.worksetNodeState(aNodeState, tNodeStateWS);

        // Workset control
        using ControlScalar = typename LocalJacobian::ControlScalarType;
        Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", mNumCells, mNumNodesPerCell);
        mWorksetBase.worksetControl(aControl, tControlWS);

        // Workset Jacobian wrt current local states
        using JacobianScalar = typename LocalJacobian::ResultScalarType;
        Plato::ScalarMultiVectorT<JacobianScalar> tJacobianWS("Jacobian Local State Workset", mNumCells, mNumLocalDofsPerCell);

        // Call evaluate function - compute Jacobian wrt the current local states
        mGlobalVecFuncJacobianC->evaluate(tGlobalStateWS, tPrevGlobalStateWS, tLocalStateWS, tPrevLocalStateWS,
                                          tNodeStateWS, tControlWS, tConfigWS, tJacobianWS, aTimeStep);

        return tJacobianWS;
    }

    /**************************************************************************//**
    * \brief Compute Jacobian wrt previous local state of the global residual
    * \param [in] aGlobalState global state at current time step
    * \param [in] aPrevGlobalState global state at previous time step
    * \param [in] aLocalState local state at current time step
    * \param [in] aPrevLocalState local state at previous time step
    * \param [in] aControl control parameters
    * \param [in] aNodeState pressure gradient
    * \param [in] aTimeStep time step
    * \return Jacobian wrt previous local state of the global residual
    ******************************************************************************/
    Plato::ScalarMultiVectorT<typename LocalJacobianP::ResultScalarType>
    gradient_cp(const Plato::ScalarVector & aGlobalState,
                const Plato::ScalarVector & aPrevGlobalState,
                const Plato::ScalarVector & aLocalState,
                const Plato::ScalarVector & aPrevLocalState,
                const Plato::ScalarVector & aNodeState,
                const Plato::ScalarVector & aControl,
                Plato::Scalar aTimeStep = 0.0) const
    {
        // Workset config
        using ConfigScalar = typename LocalJacobianP::ConfigScalarType;
        Plato::ScalarArray3DT<ConfigScalar>
            tConfigWS("Config Workset", mNumCells, mNumNodesPerCell, mNumSpatialDims);
        mWorksetBase.worksetConfig(tConfigWS);

        // Workset current global state
        using GlobalStateScalar = typename LocalJacobianP::StateScalarType;
        Plato::ScalarMultiVectorT<GlobalStateScalar>
            tGlobalStateWS("Current Global State Workset", mNumCells, mNumGlobalDofsPerCell);
        mWorksetBase.worksetState(aGlobalState, tGlobalStateWS);

        // Workset previous global state
        using PrevGlobalStateScalar = typename LocalJacobianP::PrevStateScalarType;
        Plato::ScalarMultiVectorT<PrevGlobalStateScalar>
            tPrevGlobalStateWS("Previous Global State Workset", mNumCells, mNumGlobalDofsPerCell);
        mWorksetBase.worksetState(aPrevGlobalState, tPrevGlobalStateWS);

        // Workset local state
        using LocalStateScalar = typename LocalJacobianP::LocalStateScalarType;
        Plato::ScalarMultiVectorT<LocalStateScalar>
            tLocalStateWS("Local Current State Workset", mNumCells, mNumLocalDofsPerCell);
        mWorksetBase.worksetLocalState(aLocalState, tLocalStateWS);

        // Workset previous local state
        using PrevLocalStateScalar = typename LocalJacobianP::PrevLocalStateScalarType;
        Plato::ScalarMultiVectorT<PrevLocalStateScalar>
            tPrevLocalStateWS("Previous Local State Workset", mNumCells, mNumLocalDofsPerCell);
        mWorksetBase.worksetLocalState(aPrevLocalState, tPrevLocalStateWS);

        // Workset node state, i.e. projected pressure gradient
        using NodeStateScalar = typename LocalJacobianP::NodeStateScalarType;
        Plato::ScalarMultiVectorT<NodeStateScalar> tNodeStateWS("Node State Workset", mNumCells, mNumNodeStatePerCell);
        mWorksetBase.worksetNodeState(aNodeState, tNodeStateWS);

        // Workset control
        using ControlScalar = typename LocalJacobianP::ControlScalarType;
        Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", mNumCells, mNumNodesPerCell);
        mWorksetBase.worksetControl(aControl, tControlWS);

        // Workset Jacobian wrt previous local states
        using JacobianScalar = typename LocalJacobianP::ResultScalarType;
        Plato::ScalarMultiVectorT<JacobianScalar> tJacobianWS("Jacobian Local State Workset", mNumCells, mNumLocalDofsPerCell);

        // Call evaluate function - compute Jacobian wrt the previous local states
        mGlobalVecFuncJacobianCP->evaluate(tGlobalStateWS, tPrevGlobalStateWS, tLocalStateWS, tPrevLocalStateWS,
                                           tNodeStateWS, tControlWS, tConfigWS, tJacobianWS, aTimeStep);

        return tJacobianWS;
    }
};
// class VectorFunctionVMSInc





template<typename SimplexPhysics>
class PlasticityProblem : public Plato::AbstractProblem
{
private:
    static constexpr auto mSpatialDim = SimplexPhysics::mNumSpatialDims; /*!< spatial dimensions */

    // Required
    Plato::VectorFunctionVMSInc<SimplexPhysics> mGlobalResidual;  /*!< global equality constraint interface */
    Plato::LocalVectorFunctionInc<SimplexPhysics> mLocalResidual; /*!< local equality constraint interface */
    Plato::VectorFunctionVMS<Plato::StabilizedMechanics<mSpatialDim>::ProjectorT> mProjectResidual; /*!< global pressure gradient projection interface */
};
// class PlasticityProblem

}
// namespace Plato

namespace ElastoPlasticityTest
{

}
// ElastoPlasticityTest
