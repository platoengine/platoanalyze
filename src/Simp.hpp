#pragma once

#include <stdio.h>
#include <Teuchos_ParameterList.hpp>

#include "AnalyzeMacros.hpp"
#include "PlatoStaticsTypes.hpp"

namespace Plato
{

/******************************************************************************//**
 * \brief Modified Solid Isotropic Material Penalization (MSIMP) model
**********************************************************************************/
class MSIMP
{
private:
    Plato::Scalar mMinValue;                  /*!< minimum ersatz material */
    Plato::Scalar mPenaltyParam;              /*!< penalty parameter */
    Plato::Scalar mUpperBoundOnPenaltyParam;  /*!< continuation parameter: upper bound on penalty parameter */
    Plato::Scalar mAdditiveContinuationValue; /*!< continuation parameter: multiplier on penalty parameter */

public:
    /******************************************************************************//**
     * \brief Constructor
     * \param [in] aPenalty penalty parameter
     * \param [in] aMinValue minimum ersatz material
    **********************************************************************************/
    explicit MSIMP(const Plato::Scalar & aPenalty, const Plato::Scalar & aMinValue) :
            mMinValue(aMinValue),
            mPenaltyParam(aPenalty),
            mUpperBoundOnPenaltyParam(3.0),
            mAdditiveContinuationValue(0.1)
    {
    }

    /******************************************************************************//**
     * \brief Constructor
     * \param [in] aInputParams input parameters
    **********************************************************************************/
    explicit MSIMP(Teuchos::ParameterList & aInputParams)
    {
        mPenaltyParam = aInputParams.get<Plato::Scalar>("Exponent", 3.0);
        mMinValue = aInputParams.get<Plato::Scalar>("Minimum Value", 1e-9);
        mUpperBoundOnPenaltyParam = aInputParams.get<Plato::Scalar>("Penalty Exponent Upper Bound", 3.0);
        mAdditiveContinuationValue = aInputParams.get<Plato::Scalar>("Additive Continuation", 0.1);
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
        tMsg << "Modified Solid Isotropic Material Penalization Model: New penalty parameter is set to '"
                << mPenaltyParam << "'. Previous penalty parameter was '" << tPreviousPenalty << "'.\n";
        REPORT(tMsg.str().c_str())
    }

    /******************************************************************************//**
     * \brief Set SIMP model parameters
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
     * \brief Set SIMP model penalty
     * \param [in] aInput penalty
    **********************************************************************************/
    void setPenalty(const Plato::Scalar & aInput)
    {
        mPenaltyParam = aInput;
    }

    /******************************************************************************//**
     * \brief Set minimum ersatz material value
     * \param [in] aInput minimum value
    **********************************************************************************/
    void setMinimumErsatzMaterial(const Plato::Scalar & aInput)
    {
        mMinValue = aInput;
    }

    /******************************************************************************//**
     * \brief Set additive continuation value
     * \param [in] aInput additive continuation value
    **********************************************************************************/
    void setAdditiveContinuationValue(const Plato::Scalar & aInput)
    {
        mAdditiveContinuationValue = aInput;
    }

    /******************************************************************************//**
     * \brief Constructor
     * \param [in] aInputParams input parameters
     * \return penalized ersatz material
    **********************************************************************************/
    template<typename ScalarType>
    DEVICE_TYPE inline ScalarType operator()( const ScalarType & aInput ) const
    {
        auto tOutput = mMinValue + ( (static_cast<ScalarType>(1.0) - mMinValue) * pow(aInput, mPenaltyParam) );
        return tOutput;
    }
};
// class MSIMP

} // namespace Plato
