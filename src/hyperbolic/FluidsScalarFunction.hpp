/*
 * FluidsScalarFunction.hpp
 *
 *  Created on: Apr 6, 2021
 */

#pragma once

#include "UtilsTeuchos.hpp"
#include "SpatialModel.hpp"

#include "hyperbolic/FluidsWorkSetsUtils.hpp"
#include "hyperbolic/FluidsCriterionBase.hpp"
#include "hyperbolic/SimplexFluidsFadTypes.hpp"
#include "hyperbolic/AbstractScalarFunction.hpp"

namespace Plato
{

namespace Fluids
{

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
    (const Plato::SpatialModel    & aModel,
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

}
// namespace Fluids

}
// namespace Plato
