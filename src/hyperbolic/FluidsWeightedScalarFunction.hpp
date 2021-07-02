/*
 * FluidsWeightedScalarFunction.hpp
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

}
// namespace Fluids

}
// namespace Plato

#include "hyperbolic/IncompressibleFluids.hpp"

#ifdef PLATOANALYZE_1D
extern template class Plato::Fluids::WeightedScalarFunction<Plato::IncompressibleFluids<1>>;
#endif

#ifdef PLATOANALYZE_2D
extern template class Plato::Fluids::WeightedScalarFunction<Plato::IncompressibleFluids<2>>;
#endif

#ifdef PLATOANALYZE_3D
extern template class Plato::Fluids::WeightedScalarFunction<Plato::IncompressibleFluids<3>>;
#endif
