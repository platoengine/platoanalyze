
#ifndef RAMP_HPP
#define RAMP_HPP

#include <Teuchos_ParameterList.hpp>

#include "PlatoStaticsTypes.hpp"

namespace Plato
{

/******************************************************************************//**
 * \brief Rational Approximation of Material Properties (RAMP) model class
**********************************************************************************/
class RAMP
{
    Plato::Scalar mMinValue;                   /*!< minimum Ersatz material */
    Plato::Scalar mPenaltyParam;               /*!< penalty parameter */
    Plato::Scalar mUpperBoundOnPenaltyParam;   /*!< continuation parameter: upper bound on penalty parameter */
    Plato::Scalar mAdditiveContinuationValue;  /*!< additive continuation parameter */

public:
    /******************************************************************************//**
     * \brief RAMP model default constructor
    **********************************************************************************/
    RAMP() :
            mMinValue(0),
            mPenaltyParam(3),
            mUpperBoundOnPenaltyParam(3.0),
            mAdditiveContinuationValue(0.1)
    {
    }

    /******************************************************************************//**
     * \brief RAMP model constructor
     * \param [in] aParamList parameter list
    **********************************************************************************/
    RAMP(Teuchos::ParameterList & aParamList)
    {
        mPenaltyParam = aParamList.get < Plato::Scalar > ("Exponent", 3.0);
        mMinValue = aParamList.get < Plato::Scalar > ("Minimum Value", 0.0);
        mAdditiveContinuationValue = aParamList.get<Plato::Scalar>("Additive Continuation", 0.1);
        mUpperBoundOnPenaltyParam = aParamList.get<Plato::Scalar>("Penalty Exponent Upper Bound", 3.0);
    }

    /******************************************************************************//**
     * \brief Update penalty model parameters within a frequency of optimization iterations
    **********************************************************************************/
    void update()
    {
        // update SIMP penalty parameter
        auto tPreviousPenalty = mPenaltyParam;
        auto tSuggestedPenalty = tPreviousPenalty + mAdditiveContinuationValue;
        mPenaltyParam = tSuggestedPenalty >= mUpperBoundOnPenaltyParam ? mUpperBoundOnPenaltyParam : tSuggestedPenalty;
        std::ostringstream tMsg;
        tMsg << "Rational Approximation of Material Properties Model: New penalty parameter is set to '"
                << mPenaltyParam << "'. Previous penalty parameter was '" << tPreviousPenalty << "'.\n";
        REPORT(tMsg.str().c_str())
    }

    /******************************************************************************//**
     * \brief Set RAMP model parameters
     * \param [in] aInput parameter list
    **********************************************************************************/
    void setParameters(const std::map<std::string, Plato::Scalar>& aInputs)
    {
        auto tParamMapIterator = aInputs.find("Exponent");
        if(tParamMapIterator == aInputs.end())
        {
            mPenaltyParam = 3.0; // default value
        }
        mPenaltyParam = tParamMapIterator->second;

        tParamMapIterator = aInputs.find("Minimum Value");
        if(tParamMapIterator == aInputs.end())
        {
            mMinValue = 1e-9; // default value
        }
        mMinValue = tParamMapIterator->second;

        tParamMapIterator = aInputs.find("Additive Continuation");
        if(tParamMapIterator == aInputs.end())
        {
            mAdditiveContinuationValue = 0.1; // default value
        }
        mAdditiveContinuationValue = tParamMapIterator->second;
    }

    /******************************************************************************//**
     * \brief Evaluate RAMP model
     * \param [in] aInput material density
    **********************************************************************************/
    template<typename ScalarType>
    DEVICE_TYPE inline ScalarType operator()(ScalarType aInput) const
    {
        ScalarType tOutput = mMinValue
                + (static_cast<ScalarType>(1.0) - mMinValue) * aInput / (static_cast<ScalarType>(1.0)
                        + mPenaltyParam * (static_cast<ScalarType>(1.0) - aInput));
        return (tOutput);
    }
};

}

#endif
