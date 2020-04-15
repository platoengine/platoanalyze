#pragma once

#ifndef T_PI
#define T_PI 3.1415926535897932385
#endif

#include <stdio.h>
#include <Teuchos_ParameterList.hpp>

#include "PlatoStaticsTypes.hpp"

namespace Plato
{

/******************************************************************************//**
 * \brief Heaviside penalty model model class
**********************************************************************************/
class Heaviside
{
    Plato::Scalar mPenaltyParam;              /*!< penalty parameter */
    Plato::Scalar mRegLength;                 /*!< regularization length */
    Plato::Scalar mMinValue;                  /*!< minimum Ersatz material */
    Plato::Scalar mUpperBoundOnPenaltyParam;  /*!< continuation parameter: upper bound on penalty parameter */
    Plato::Scalar mAdditiveContinuationValue; /*!< additive continuation parameter */

public:
    /******************************************************************************//**
     * \brief Default Heaviside model constructor
     * \param [in] aPenalty   penalty parameter (default = 1.0)
     * \param [in] aRegLength regularization length (default = 1.0)
     * \param [in] aMinValue  minimum Ersatz material constant (default = 0.0)
    **********************************************************************************/
    Heaviside(Plato::Scalar aPenalty = 1.0, Plato::Scalar aRegLength = 1.0, Plato::Scalar aMinValue = 0.0) :
            mPenaltyParam(aPenalty),
            mRegLength(aRegLength),
            mMinValue(aMinValue),
            mUpperBoundOnPenaltyParam(3.0),
            mAdditiveContinuationValue(0.1)
    {
    }

    /******************************************************************************//**
     * \brief Heaviside model constructor
     * \param [in] aParamList parameter list
    **********************************************************************************/
    Heaviside(Teuchos::ParameterList & aParamList)
    {
        mPenaltyParam = aParamList.get<Plato::Scalar>("Exponent", 1.0);
        mRegLength = aParamList.get<Plato::Scalar>("Regularization Length", 1.0);
        mMinValue = aParamList.get<Plato::Scalar>("Minimum Value", 0.0);
        mAdditiveContinuationValue = aParamList.get<Plato::Scalar>("Additive Continuation", 0.1);
        mUpperBoundOnPenaltyParam = aParamList.get<Plato::Scalar>("Penalty Exponent Upper Bound", 1.0);
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
        tMsg << "Heaviside Penalization Model: New penalty parameter is set to '" << mPenaltyParam
                << "'. Previous penalty parameter was '" << tPreviousPenalty << "'.\n";
        REPORT(tMsg.str().c_str())
    }

    /******************************************************************************//**
     * \brief Set Heaviside model parameters
     * \param [in] aInput parameter list
    **********************************************************************************/
    void setParameters(const std::map<std::string, Plato::Scalar>& aInputs)
    {
        auto tParamMapIterator = aInputs.find("Exponent");
        if(tParamMapIterator == aInputs.end())
        {
            mPenaltyParam = 1.0; // default value
        }
        mPenaltyParam = tParamMapIterator->second;

        tParamMapIterator = aInputs.find("Minimum Value");
        if(tParamMapIterator == aInputs.end())
        {
            mMinValue = 1e-9; // default value
        }
        mMinValue = tParamMapIterator->second;

        tParamMapIterator = aInputs.find("Regularization Length");
        if(tParamMapIterator == aInputs.end())
        {
            mMinValue = 1.0; // default value
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
     * \brief Evaluate Heaviside model
     * \param [in] aInput material density
    **********************************************************************************/
    template<typename ScalarType>
    DEVICE_TYPE inline ScalarType operator()( ScalarType aInput ) const
    {
        if (aInput <= -mRegLength)
        {
            return mMinValue;
        }
        else
        if (aInput >=  mRegLength)
        {
            return 1.0;
        }
        else
        {
            return mMinValue + (1.0 - mMinValue) * pow(1.0/2.0*(1.0 + sin(T_PI*aInput/(2.0*mRegLength))),mPenaltyParam);
        }
    }
};
// class Heaviside

} // namespace Plato
