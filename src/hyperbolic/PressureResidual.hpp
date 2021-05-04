/*
 * PressureResidual.hpp
 *
 *  Created on: Apr 7, 2021
 */

#pragma once

#include <Teuchos_ParameterList.hpp>

#include "BLAS2.hpp"
#include "MetaData.hpp"
#include "WorkSets.hpp"
#include "NaturalBCs.hpp"
#include "SpatialModel.hpp"
#include "ExpInstMacros.hpp"
#include "AbstractVolumeIntegrand.hpp"
#include "LinearTetCubRuleDegreeOne.hpp"

#include "hyperbolic/SimplexFluids.hpp"
#include "hyperbolic/MassConservationUtils.hpp"
#include "hyperbolic/MomentumSurfaceForces.hpp"
#include "hyperbolic/SimplexFluidsFadTypes.hpp"
#include "hyperbolic/AbstractVectorFunction.hpp"

namespace Plato
{

namespace Fluids
{

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

}
// namespace Fluids

}
// namespace Plato

#include "hyperbolic/IncompressibleFluids.hpp"

#ifdef PLATOANALYZE_1D
PLATO_EXPL_DEC_FLUIDS(Plato::Fluids::PressureResidual, Plato::MassConservation, Plato::SimplexFluids, 1, 1)
#endif

#ifdef PLATOANALYZE_2D
PLATO_EXPL_DEC_FLUIDS(Plato::Fluids::PressureResidual, Plato::MassConservation, Plato::SimplexFluids, 2, 1)
#endif

#ifdef PLATOANALYZE_3D
PLATO_EXPL_DEC_FLUIDS(Plato::Fluids::PressureResidual, Plato::MassConservation, Plato::SimplexFluids, 3, 1)
#endif
