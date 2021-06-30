/*
 * CriterionThermalCompliance.hpp
 *
 *  Created on: June 16, 2021
 */

#pragma once

#include "BLAS2.hpp"
#include "NaturalBCs.hpp"
#include "SpatialModel.hpp"
#include "UtilsTeuchos.hpp"
#include "ExpInstMacros.hpp"
#include "LinearTetCubRuleDegreeOne.hpp"

#include "hyperbolic/FluidsUtils.hpp"
#include "hyperbolic/SimplexFluids.hpp"
#include "hyperbolic/FluidsThermalSources.hpp"
#include "hyperbolic/SimplexFluidsFadTypes.hpp"
#include "hyperbolic/AbstractScalarFunction.hpp"
#include "hyperbolic/EnergyConservationUtils.hpp"

namespace Plato
{

namespace Fluids
{

/***************************************************************************//**
 * \tparam PhysicsT    Plato physics type
 * \tparam EvaluationT Forward Automatic Differentiation (FAD) evaluation type
 *
 * \class CriterionThermalCompliance
 *
 * \brief Evaluate the thermal compliance of a body, defined as
 *
 *  \f[ \int_{\Omega_e} w^h Q_n d\Omega_e + \int_{\Gamma_e} w^h F_n d\Gamma_e \f],
 *
 * where the subscript \f$ n \f$ is the current time step, \f$ Q \f$ is a 
 * thermal source and \f$ F \f$ is the thermal flux.
 ******************************************************************************/
template<typename PhysicsT, typename EvaluationT>
class CriterionThermalCompliance : public Plato::Fluids::AbstractScalarFunction<PhysicsT, EvaluationT>
{
private:
    static constexpr auto mNumSpatialDims  = PhysicsT::SimplexT::mNumSpatialDims; /*!< number of spatial dimensions */
    static constexpr auto mNumNodesPerCell = PhysicsT::SimplexT::mNumNodesPerCell; /*!< number of nodes per cell/element */
    static constexpr auto mNumTempDofsPerNode = PhysicsT::mNumEnergyDofsPerNode; /*!< number of degrees of freedom per node */
    static constexpr auto mNumTempDofsPerCell = PhysicsT::mNumEnergyDofsPerCell; /*!< number of degrees of freedom per cell */

    // set local ad type
    using ResultT   = typename EvaluationT::ResultScalarType; /*!< result FAD evaluation type */
    using ConfigT   = typename EvaluationT::ConfigScalarType; /*!< configuration FAD evaluation type */
    using ControlT  = typename EvaluationT::ControlScalarType; /*!< control FAD evaluation type */
    using CurTempT  = typename EvaluationT::CurrentEnergyScalarType; /*!< current temperature FAD evaluation type */

    // member data
    std::string mFuncName; /*!< scalar funciton name */

    // member metadata/database
    Plato::DataMap& mDataMap; /*!< holds output database */
    const Plato::SpatialDomain& mSpatialDomain; /*!< holds mesh and entity sets metadata for a domain (i.e. element block) */

    // member evaluators
    Plato::LinearTetCubRuleDegreeOne<mNumSpatialDims> mCubatureRule; /*!< cubature integration rule */
    Plato::Fluids::ThermalSources<PhysicsT, EvaluationT> mThermalSources; /*!< interface to thermal source evaluator */
    std::shared_ptr<Plato::NaturalBCs<mNumSpatialDims, mNumTempDofsPerNode>> mHeatFlux; /*!< natural boundary condition evaluator */

public:
    /***************************************************************************//**
     * \brief Constructor
     * \param [in] aName    scalar function name
     * \param [in] aDomain  spatial domain metadata
     * \param [in] aDataMap output database
     * \param [in] aInputs  input database
     ******************************************************************************/
    CriterionThermalCompliance
    (const std::string          & aName,
     const Plato::SpatialDomain & aDomain,
     Plato::DataMap             & aDataMap,
     Teuchos::ParameterList     & aInputs) :
         mDataMap(aDataMap),
         mSpatialDomain(aDomain),
         mFuncName(aName)
    {
        this->initializeThermalFluxes(aInputs);
        mThermalSources.initializeThermalSources(aDomain, aDataMap, aInputs);
    }

    /***************************************************************************//**
     * \brief Destructor
     ******************************************************************************/
    virtual ~CriterionThermalCompliance(){}

    /***************************************************************************//**
     * \fn std::string name
     * \brief Returns scalar function name
     * \return scalar function name
     ******************************************************************************/
    std::string name() const override { return mFuncName; }

    /***************************************************************************//**
     * \fn void evaluate
     * \brief Evaluate scalar function within the computational domain \f$ \Omega \f$.
     * \param [in] aWorkSets holds state work sets initialize with correct FAD types
     * \param [in] aResult   1D output work set of size number of cells
     ******************************************************************************/
    void evaluate
    (const Plato::WorkSets &aWorkSets, 
     Plato::ScalarVectorT<ResultT> &aResultWS) const
    {
        if(mThermalSources.empty()) { return; }
        
        auto tNumCells = mSpatialDomain.numCells();
        if( tNumCells != static_cast<Plato::OrdinalType>(aResultWS.extent(0)) )
        {
            THROWERR(std::string("Number of elements mismatch. Spatial domain and output/result workset ")
                + "cell number does not match. " + "Spatial domain has '" + std::to_string(tNumCells)
                + "' cells/elements and output workset has '" + std::to_string(aResultWS.extent(0)) + "' cells/elements.")
        }

        // set current state worksets
        auto tCurTempWS = Plato::metadata<Plato::ScalarMultiVectorT<CurTempT>>(aWorkSets.get("current temperature"));

        // evaluate thermal sources
        Plato::ScalarMultiVectorT<ResultT> tThermalSource("thermal source", tNumCells, mNumNodesPerCell);
        mThermalSources.evaluate(aWorkSets, tThermalSource);

        // calculate inner product between current temperature and thermal source worksets    
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
        {               
            for(Plato::OrdinalType tDof = 0; tDof < mNumTempDofsPerCell; tDof++)
            {
                aResultWS(aCellOrdinal) += tCurTempWS(aCellOrdinal, tDof) * tThermalSource(aCellOrdinal, tDof);
            }
        }, "calculate inner product between current temperature and thermal source worksets");
    }

    /***************************************************************************//**
     * \fn void evaluateBoundary
     * \brief Evaluate scalar function along the computational boudary \f$ \Gamma \f$.
     * \param [in] aSpatialModel spatial domain metadata, holds mesh and entity sets
     * \param [in] aWorkSets     state work sets initialized with proper FAD types
     * \param [in] aResult       output work set 
     ******************************************************************************/
    void evaluateBoundary
    (const Plato::SpatialModel & aSpatialModel, 
     const Plato::WorkSets & aWorkSets, 
     Plato::ScalarVectorT<ResultT> & aResultWS) 
     const override
    {
        if( mHeatFlux != nullptr )
        {
            // get state worksets
            auto tConfigWS  = Plato::metadata<Plato::ScalarArray3DT<ConfigT>>(aWorkSets.get("configuration"));
            auto tControlWS = Plato::metadata<Plato::ScalarMultiVectorT<ControlT>>(aWorkSets.get("control"));
            auto tCurTempWS = Plato::metadata<Plato::ScalarMultiVectorT<CurTempT>>(aWorkSets.get("current temperature"));

            // evaluate prescribed flux
            auto tNumCells = aResultWS.extent(0);
            Plato::ScalarMultiVectorT<ResultT> tHeatFluxWS("heat flux", tNumCells, mNumTempDofsPerCell);
            mHeatFlux->get( aSpatialModel, tCurTempWS, tControlWS, tConfigWS, tHeatFluxWS );

            // inner product
            Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
            {
                for(Plato::OrdinalType tDof = 0; tDof < mNumTempDofsPerCell; tDof++)
                {
                    aResultWS(aCellOrdinal) += tCurTempWS(aCellOrdinal, tDof) * tHeatFluxWS(aCellOrdinal, tDof);
                }
            }, "calculate inner product between current temperature and thermal flux worksets");
        }
    }

private:
    /***************************************************************************//**
     * \brief Initialize thermal fluxes.
     * \param [in] aInputs  input database
     ******************************************************************************/
    void initializeThermalFluxes(Teuchos::ParameterList& aInputs)
    {
        if(aInputs.isSublist("Thermal Natural Boundary Conditions"))
        {
            auto tSublist = aInputs.sublist("Thermal Natural Boundary Conditions");
            mHeatFlux = std::make_shared<Plato::NaturalBCs<mNumSpatialDims, mNumTempDofsPerNode>>(tSublist);
        }
    }
};
// class CriterionThermalCompliance

}
// namespace Fluids

}
// namespace Plato

#include "hyperbolic/IncompressibleFluids.hpp"

#ifdef PLATOANALYZE_1D
PLATO_EXPL_DEC_FLUIDS(Plato::Fluids::CriterionThermalCompliance, Plato::IncompressibleFluids, Plato::SimplexFluids, 1, 1)
#endif

#ifdef PLATOANALYZE_2D
PLATO_EXPL_DEC_FLUIDS(Plato::Fluids::CriterionThermalCompliance, Plato::IncompressibleFluids, Plato::SimplexFluids, 2, 1)
#endif

#ifdef PLATOANALYZE_3D
PLATO_EXPL_DEC_FLUIDS(Plato::Fluids::CriterionThermalCompliance, Plato::IncompressibleFluids, Plato::SimplexFluids, 3, 1)
#endif
