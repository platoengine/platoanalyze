/*
 * FluidsUniformThermalSource.hpp
 *
 *  Created on: June 16, 2021
 */

#pragma once

#include <Teuchos_ParameterList.hpp>

#include "BLAS2.hpp"
#include "MetaData.hpp"
#include "WorkSets.hpp"
#include "SpatialModel.hpp"
#include "UtilsTeuchos.hpp"
#include "ExpInstMacros.hpp"
#include "InterpolateFromNodal.hpp"
#include "AbstractVolumetricSource.hpp"
#include "LinearTetCubRuleDegreeOne.hpp"

#include "hyperbolic/FluidsUtils.hpp"
#include "hyperbolic/SimplexFluids.hpp"
#include "hyperbolic/SimplexFluidsFadTypes.hpp"
#include "hyperbolic/EnergyConservationUtils.hpp"

namespace Plato
{

namespace Fluids
{

namespace SIMP
{

template<typename PhysicsT, typename EvaluationT>
class UniformThermalSource : public Plato::AbstractVolumetricSource<PhysicsT, EvaluationT>
{
private:
    static constexpr auto mNumSpatialDims     = PhysicsT::SimplexT::mNumSpatialDims; /*!< number of spatial dimensions */
    static constexpr auto mNumNodesPerCell    = PhysicsT::SimplexT::mNumNodesPerCell; /*!< number of nodes per cell/element */
    static constexpr auto mNumTempDofsPerCell = PhysicsT::mNumEnergyDofsPerCell; /*!< number of degrees of freedom per cell */

    // set local ad type
    using ResultT   = typename EvaluationT::ResultScalarType; /*!< result FAD evaluation type */
    using ConfigT   = typename EvaluationT::ConfigScalarType; /*!< configuration FAD evaluation type */
    using ControlT  = typename EvaluationT::ControlScalarType; /*!< control FAD evaluation type */

    // member parameters
    Plato::Scalar mMagnitude = 0.0; /*!< thermal source magnitude */
    Plato::Scalar mPenaltyExponent = 3.0; /*!< thermal source simp penalty model exponent */
    Plato::Scalar mThermalConductivity = 1.0; /*!< thermal conductivity */
    Plato::Scalar mCharacteristicLength = 0.0; /*!< characteristic lenght */
    Plato::Scalar mReferenceTemperature = 1.0; /*!< reference temperature */

    std::string mFuncName; /*!< scalar funciton name */
    std::string mElemBlock; /*!< element block name where thermal source is applied */

    // member metadata
    Plato::DataMap& mDataMap; /*!< holds output metadata */
    const Plato::SpatialDomain& mSpatialDomain; /*!< holds mesh and entity sets metadata for a domain (i.e. element block) */
    Plato::LinearTetCubRuleDegreeOne<mNumSpatialDims> mCubatureRule; /*!< cubature integration rule */

public:
    UniformThermalSource    
    (const std::string          & aFuncName,
     const Plato::SpatialDomain & aDomain,
     Plato::DataMap             & aDataMap,
     Teuchos::ParameterList     & aInputs) : 
        mDataMap(aDataMap),
        mSpatialDomain(aDomain),
        mFuncName(aFuncName)
    {
        this->initialize(aInputs);
    }

    virtual ~UniformThermalSource(){}
    
    std::string type() const override
    {
        return "uniform";
    }

    std::string name() const override
    {
        return mFuncName;
    }

    void evaluate
    (const Plato::WorkSets & aWorkSets, 
     Plato::ScalarMultiVectorT<ResultT> & aResultWS,
     Plato::Scalar aMultiplier = 1.0) 
     const override
    {
        if(mElemBlock.empty()) { return; }

        auto tMySpatialDomainElemBlockName = mSpatialDomain.getElementBlockName();
        if(mElemBlock == tMySpatialDomainElemBlockName)
        {
            auto tNumCells = mSpatialDomain.numCells();
            if (tNumCells != static_cast<Plato::OrdinalType>(aResultWS.extent(0)) )
            {
                THROWERR(std::string("Number of elements mismatch. Spatial domain and output/result workset ") 
                    + "cell number does not match. " + "Spatial domain has '" + std::to_string(tNumCells) 
                    + "' cells/elements and output workset has '" + std::to_string(aResultWS.extent(0)) + "' cells/elements.")
            }

            // set local functors
            Plato::ComputeGradientWorkset<mNumSpatialDims> tComputeGradient;

            // set constant thermal source
            Plato::ScalarVectorT<Plato::Scalar> tThermalSource("thermal source", tNumCells);
            Plato::blas1::fill(mMagnitude, tThermalSource);

            // set local arrays
            Plato::ScalarVectorT<ConfigT> tCellVolume("cell weight", tNumCells);
            Plato::ScalarArray3DT<ConfigT> tGradient("cell gradient", tNumCells, mNumNodesPerCell, mNumSpatialDims);

            // set input state worksets
            auto tConfigWS = Plato::metadata<Plato::ScalarArray3DT<ConfigT>>(aWorkSets.get("configuration"));
            auto tControlWS = Plato::metadata<Plato::ScalarMultiVectorT<ControlT>>(aWorkSets.get("control"));

            // transfer member data to device
            auto tRefTemp = mReferenceTemperature;
            auto tCharLength = mCharacteristicLength;
            auto tThermalCond = mThermalConductivity;
            auto tPenaltyExponent = mPenaltyExponent;

            auto tCubWeight = mCubatureRule.getCubWeight();
            auto tBasisFunctions = mCubatureRule.getBasisFunctions();
            Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType &aCellOrdinal)
            {
                tComputeGradient(aCellOrdinal, tGradient, tConfigWS, tCellVolume);
                tCellVolume(aCellOrdinal) = tCellVolume(aCellOrdinal) * tCubWeight;

                auto tDimlessConstant = aMultiplier * ( (tCharLength * tCharLength) / (tThermalCond * tRefTemp) );
                ControlT tPenalizedDimlessConstant = Plato::Fluids::penalize_heat_source_constant<mNumNodesPerCell>
                    (aCellOrdinal, tDimlessConstant, tPenaltyExponent, tControlWS);
                Plato::Fluids::integrate_scalar_field<mNumTempDofsPerCell>
                    (aCellOrdinal, tBasisFunctions, tCellVolume, tThermalSource, aResultWS, -tPenalizedDimlessConstant);
            },"intergate thermal source term");
        }
    }

private:
    /***************************************************************************//**
     * \brief Initialize thermal source.
     * \param [in] aInputs  input database
     ******************************************************************************/
    void initialize(Teuchos::ParameterList& aInputs)
    {
        if( aInputs.isSublist("Thermal Sources") )
        {
            auto tThermalSourceParamList = aInputs.sublist("Thermal Sources");
            mMagnitude = Plato::teuchos::parse_parameter<Plato::Scalar>("Value", mFuncName, tThermalSourceParamList);
            mElemBlock = Plato::teuchos::parse_parameter<std::string>("Element Block", mFuncName, tThermalSourceParamList);

            this->initializeMaterialProperties(aInputs);
        }
    }

    /***************************************************************************//**
     * \brief Initialize material proerties.
     * \param [in] aInputs  input database
     ******************************************************************************/
    void initializeMaterialProperties(Teuchos::ParameterList& aInputs)
    {
        auto tMaterialName = mSpatialDomain.getMaterialName();
        mThermalConductivity = Plato::Fluids::get_material_property<Plato::Scalar>("Thermal Conductivity", tMaterialName, aInputs);
        Plato::is_positive_finite_number(mThermalConductivity, "Thermal Conductivity");

        mCharacteristicLength = Plato::Fluids::get_material_property<Plato::Scalar>("Characteristic Length", tMaterialName, aInputs);
        Plato::is_positive_finite_number(mCharacteristicLength, "Characteristic Length");

        mReferenceTemperature = Plato::Fluids::get_material_property<Plato::Scalar>("Reference Temperature", tMaterialName, aInputs);
        if(mReferenceTemperature == static_cast<Plato::Scalar>(0.0))
            { THROWERR(std::string("'Reference Temperature' keyword cannot be set to zero.")) }
            
        if(Plato::Fluids::is_material_property_defined("Thermal Source Penalty Exponent", tMaterialName, aInputs))
        {
            mPenaltyExponent = Plato::Fluids::get_material_property<Plato::Scalar>("Thermal Source Penalty Exponent", tMaterialName, aInputs);
            Plato::is_positive_finite_number(mPenaltyExponent, "Thermal Source Penalty Exponent");
        }
    }
};
// class UniformThermalSource

}
// namespace SIMP

template<typename PhysicsT, typename EvaluationT>
class UniformThermalSource : public Plato::AbstractVolumetricSource<PhysicsT, EvaluationT>
{
private:
    static constexpr auto mNumSpatialDims  = PhysicsT::SimplexT::mNumSpatialDims; /*!< number of spatial dimensions */
    static constexpr auto mNumNodesPerCell = PhysicsT::SimplexT::mNumNodesPerCell; /*!< number of nodes per cell/element */
    static constexpr auto mNumTempDofsPerCell = PhysicsT::mNumEnergyDofsPerCell; /*!< number of degrees of freedom per cell */

    // set local ad type
    using ResultT   = typename EvaluationT::ResultScalarType; /*!< result FAD evaluation type */
    using ConfigT   = typename EvaluationT::ConfigScalarType; /*!< configuration FAD evaluation type */

    // member parameters
    Plato::Scalar mMagnitude = 0.0; /*!< thermal source magnitude */
    Plato::Scalar mThermalConductivity = 1.0; /*!< thermal conductivity */
    Plato::Scalar mCharacteristicLength = 0.0; /*!< characteristic lenght */
    Plato::Scalar mReferenceTemperature = 1.0; /*!< reference temperature */

    std::string mFuncName; /*!< scalar funciton name */
    std::string mElemBlock; /*!< block name where thermal source is applied */

    // member metadata
    Plato::DataMap& mDataMap; /*!< holds output metadata */
    const Plato::SpatialDomain& mSpatialDomain; /*!< holds mesh and entity sets metadata for a domain (i.e. element block) */
    Plato::LinearTetCubRuleDegreeOne<mNumSpatialDims> mCubatureRule; /*!< cubature integration rule */

public:
    UniformThermalSource    
    (const std::string          & aFuncName,
     const Plato::SpatialDomain & aDomain,
     Plato::DataMap             & aDataMap,
     Teuchos::ParameterList     & aInputs) : 
        mDataMap(aDataMap),
        mSpatialDomain(aDomain),
        mFuncName(aFuncName)
    {
        this->initialize(aInputs);
    }

    virtual ~UniformThermalSource(){}
    
    std::string type() const override
    {
        return "uniform";
    }

    std::string name() const override
    {
        return mFuncName;
    }

    void evaluate
    (const Plato::WorkSets & aWorkSets, 
     Plato::ScalarMultiVectorT<ResultT> & aResultWS,
     Plato::Scalar aMultiplier = 1.0) 
     const override
    {
        if(mElemBlock.empty()) { return; }

        auto tMySpatialDomainElemBlockName = mSpatialDomain.getElementBlockName();
        if(mElemBlock == tMySpatialDomainElemBlockName)
        {
            auto tNumCells = mSpatialDomain.numCells();
            if (tNumCells != static_cast<Plato::OrdinalType>(aResultWS.extent(0)) )
            {
                THROWERR(std::string("Number of elements mismatch. Spatial domain and output/result workset ") 
                    + "cell number does not match. " + "Spatial domain has '" + std::to_string(tNumCells) 
                    + "' cells/elements and output workset has '" + std::to_string(aResultWS.extent(0)) + "' cells/elements.")
            }

            // set local functors
            Plato::ComputeGradientWorkset<mNumSpatialDims> tComputeGradient;

            // set constant thermal source
            Plato::ScalarVectorT<Plato::Scalar> tThermalSource("thermal source", tNumCells);
            Plato::blas1::fill(mMagnitude, tThermalSource);

            // set local arrays
            Plato::ScalarVectorT<ConfigT> tCellVolume("cell weight", tNumCells);
            Plato::ScalarArray3DT<ConfigT> tGradient("cell gradient", tNumCells, mNumNodesPerCell, mNumSpatialDims);

            // set input state worksets
            auto tConfigWS = Plato::metadata<Plato::ScalarArray3DT<ConfigT>>(aWorkSets.get("configuration"));

            // transfer member data to device
            auto tRefTemp = mReferenceTemperature;
            auto tCharLength = mCharacteristicLength;
            auto tThermalCond = mThermalConductivity;

            auto tCubWeight = mCubatureRule.getCubWeight();
            auto tBasisFunctions = mCubatureRule.getBasisFunctions();
            Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType &aCellOrdinal)
            {
                tComputeGradient(aCellOrdinal, tGradient, tConfigWS, tCellVolume);
                tCellVolume(aCellOrdinal) = tCellVolume(aCellOrdinal) * tCubWeight;

                auto tDimlessConstant = aMultiplier * ( (tCharLength * tCharLength) / (tThermalCond * tRefTemp) );
                Plato::Fluids::integrate_scalar_field<mNumTempDofsPerCell>
                    (aCellOrdinal, tBasisFunctions, tCellVolume, tThermalSource, aResultWS, -tDimlessConstant);
            },"intergate thermal source term");
        }
    }

private:
    /***************************************************************************//**
     * \brief Initialize thermal source.
     * \param [in] aInputs  input database
     ******************************************************************************/
    void initialize(Teuchos::ParameterList& aInputs)
    {
        if( aInputs.isSublist("Thermal Source") )
        {
            auto tThermalSourceParamList = aInputs.sublist("Thermal Source");
            mMagnitude = Plato::teuchos::parse_parameter<Plato::Scalar>("Value", mFuncName, tThermalSourceParamList);
            mElemBlock = Plato::teuchos::parse_parameter<std::string>("Element Block", mFuncName, tThermalSourceParamList);

            this->initializeMaterialProperties(aInputs);
        }
    }

    /***************************************************************************//**
     * \brief Initialize material proerties.
     * \param [in] aInputs  input database
     ******************************************************************************/
    void initializeMaterialProperties(Teuchos::ParameterList& aInputs)
    {
        auto tMaterialName = mSpatialDomain.getMaterialName();
        mThermalConductivity = Plato::Fluids::get_material_property<Plato::Scalar>("Thermal Conductivity", tMaterialName, aInputs);
        Plato::is_positive_finite_number(mThermalConductivity, "Thermal Conductivity");

        mCharacteristicLength = Plato::Fluids::get_material_property<Plato::Scalar>("Characteristic Length", tMaterialName, aInputs);
        Plato::is_positive_finite_number(mCharacteristicLength, "Characteristic Length");

        mReferenceTemperature = Plato::Fluids::get_material_property<Plato::Scalar>("Reference Temperature", tMaterialName, aInputs);
        if(mReferenceTemperature == static_cast<Plato::Scalar>(0.0))
            { THROWERR(std::string("'Reference Temperature' keyword cannot be set to zero.")) }
    }
};
// class UniformThermalSource

}
// namespace Fluids

}
// namespace Plato

#include "hyperbolic/IncompressibleFluids.hpp"

#ifdef PLATOANALYZE_1D
PLATO_EXPL_DEC_FLUIDS(Plato::Fluids::UniformThermalSource, Plato::IncompressibleFluids, Plato::SimplexFluids, 1, 1)
PLATO_EXPL_DEC_FLUIDS(Plato::Fluids::SIMP::UniformThermalSource, Plato::IncompressibleFluids, Plato::SimplexFluids, 1, 1)
#endif

#ifdef PLATOANALYZE_2D
PLATO_EXPL_DEC_FLUIDS(Plato::Fluids::UniformThermalSource, Plato::IncompressibleFluids, Plato::SimplexFluids, 2, 1)
PLATO_EXPL_DEC_FLUIDS(Plato::Fluids::SIMP::UniformThermalSource, Plato::IncompressibleFluids, Plato::SimplexFluids, 2, 1)
#endif

#ifdef PLATOANALYZE_3D
PLATO_EXPL_DEC_FLUIDS(Plato::Fluids::UniformThermalSource, Plato::IncompressibleFluids, Plato::SimplexFluids, 3, 1)
PLATO_EXPL_DEC_FLUIDS(Plato::Fluids::SIMP::UniformThermalSource, Plato::IncompressibleFluids, Plato::SimplexFluids, 3, 1)
#endif
