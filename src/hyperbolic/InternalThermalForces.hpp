/*
 * InternalThermalForces.hpp
 *
 *  Created on: Apr 7, 2021
 */

#pragma once

#include "BLAS2.hpp"
#include "MetaData.hpp"
#include "SpatialModel.hpp"
#include "UtilsTeuchos.hpp"
#include "InterpolateFromNodal.hpp"
#include "AbstractVolumeIntegrand.hpp"
#include "LinearTetCubRuleDegreeOne.hpp"

#include "hyperbolic/FluidsUtils.hpp"
#include "hyperbolic/SimplexFluidsFadTypes.hpp"
#include "hyperbolic/EnergyConservationUtils.hpp"

// Integrand for simulation and level-set topology optimization workflows
namespace Plato
{

namespace Fluids
{

/***************************************************************************//**
 * \class InternalThermalForces
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
class InternalThermalForces : public Plato::AbstractVolumeIntegrand<PhysicsT,EvaluationT>
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
    InternalThermalForces
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
    ~InternalThermalForces(){}

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
// class InternalThermalForces

}
// namespace Fluids

}
// namespace Plato

// Integrand for density-based topology optimization
namespace Plato
{

namespace Fluids
{

namespace SIMP
{


/***************************************************************************//**
 * \class InternalThermalForces
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
class InternalThermalForces : public Plato::AbstractVolumeIntegrand<PhysicsT,EvaluationT>
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
    InternalThermalForces
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
    ~InternalThermalForces(){}

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
// class InternalThermalForces

}
// namespace SIMP

}
// namespace Fluids

}
// namespace Plato
