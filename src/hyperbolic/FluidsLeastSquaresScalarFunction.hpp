/*
 * FluidsLeastSquaresScalarFunction.hpp
 *
 *  Created on: Apr 7, 2021
 */

#pragma once

#include <Teuchos_ParameterList.hpp>

#include "BLAS1.hpp"
#include "UtilsTeuchos.hpp"
#include "SpatialModel.hpp"

#include "hyperbolic/FluidsCriterionBase.hpp"
#include "hyperbolic/FluidsCriterionFactory.hpp"

namespace Plato
{

namespace Fluids
{

/**************************************************************************//**
* \struct LeastSquaresScalarFunction
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
        mCriterionTarget = Plato::teuchos::parse_array<Plato::Scalar>("Targets", aInputs);
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

}
// namespace Plato

#include "hyperbolic/IncompressibleFluids.hpp"

#ifdef PLATOANALYZE_1D
extern template class Plato::Fluids::LeastSquaresScalarFunction<Plato::IncompressibleFluids<1>>;
#endif

#ifdef PLATOANALYZE_2D
extern template class Plato::Fluids::LeastSquaresScalarFunction<Plato::IncompressibleFluids<2>>;
#endif

#ifdef PLATOANALYZE_3D
extern template class Plato::Fluids::LeastSquaresScalarFunction<Plato::IncompressibleFluids<3>>;
#endif

