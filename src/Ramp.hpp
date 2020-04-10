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
    Plato::Scalar mMinValue;                 /*!< minimum Ersatz material */
    Plato::Scalar mPenaltyParam;             /*!< penalty parameter */
    Plato::Scalar mMultiplierOnPenaltyParam; /*!< continuation parameter: multiplier on penalty parameter */
    Plato::Scalar mUpperBoundOnPenaltyParam; /*!< continuation parameter: upper bound on penalty parameter */

public:
    /******************************************************************************//**
     * \brief RAMP model default constructor
    **********************************************************************************/
    RAMP() :
            mMinValue(0),
            mPenaltyParam(3),
            mMultiplierOnPenaltyParam(1.0),
            mUpperBoundOnPenaltyParam(3.0)
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
        mMultiplierOnPenaltyParam = aParamList.get<Plato::Scalar>("Continuation Multiplier", 1.0);
        mUpperBoundOnPenaltyParam = aParamList.get<Plato::Scalar>("Penalty Exponent Upper Bound", 3.0);
    }

    /******************************************************************************//**
     * \brief Update penalty model parameters within a frequency of optimization iterations
    **********************************************************************************/
    void update()
    {
        mPenaltyParam = mPenaltyParam >= mUpperBoundOnPenaltyParam ? mPenaltyParam : mMultiplierOnPenaltyParam * mPenaltyParam;
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
