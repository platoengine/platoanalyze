/*
 * TimeData.hpp
 *
 *  Created on: Mar 30, 2021
 */

#pragma once

#include <limits>
#include "PlatoTypes.hpp"
#include "ParseTools.hpp"

namespace Plato
{

/*******************************************************************************//**
 * \brief Struct container for simulation time parameters and utility functions
***********************************************************************************/
struct TimeData
{
public:
    Plato::OrdinalType mCurrentTimeStepIndex;  /*!< current time step index */
    Plato::OrdinalType mNumTimeSteps;          /*!< number of time steps to attempt */
    const Plato::OrdinalType mMaxNumTimeSteps; /*!< maximum number of time steps allowed */

    Plato::Scalar mCurrentTime;         /*!< current time */
    const Plato::Scalar mStartTime;     /*!< simulation start time */
    const Plato::Scalar mEndTime;       /*!< simulation end time */
    Plato::Scalar mCurrentTimeStepSize; /*!< current time step size */

    Plato::Scalar mTimeStepExpansionMultiplier; /*!< expansion multiplier for number of time steps */

    bool mMaxNumTimeStepsReached; /*!< whether the maximum number of time steps has been reached */

    /***************************************************************************//**
     * \brief Time Data constructor
     * \param [in] aInputs input parameters database
    *******************************************************************************/
    TimeData( Teuchos::ParameterList & aInputs ) :
      mCurrentTimeStepIndex(0),
      mNumTimeSteps(Plato::ParseTools::getSubParam<Plato::OrdinalType>(aInputs, "Time Stepping", "Initial Num. Pseudo Time Steps", 20)),
      mMaxNumTimeSteps(Plato::ParseTools::getSubParam<Plato::OrdinalType>(aInputs, "Time Stepping", "Maximum Num. Pseudo Time Steps", 80)),
      mCurrentTime(0.0),
      mStartTime(0.0),
      mEndTime(Plato::ParseTools::getSubParam<Plato::Scalar>(aInputs, "Time Stepping", "End Time", 1.0)),
      mCurrentTimeStepSize(mEndTime/(static_cast<Plato::Scalar>(mNumTimeSteps))),
      mTimeStepExpansionMultiplier(Plato::ParseTools::getSubParam<Plato::Scalar>(aInputs, "Time Stepping", "Expansion Multiplier", 2)),
      mMaxNumTimeStepsReached(mNumTimeSteps >= mMaxNumTimeSteps)
    {
        mNumTimeSteps = std::min(mNumTimeSteps, mMaxNumTimeSteps);
    }
    
    bool atFinalTimeStep()
    {
        Plato::Scalar tTolerance = static_cast<Plato::Scalar>(2.0) * std::numeric_limits<Plato::Scalar>::epsilon();
        return mCurrentTime > (mEndTime - tTolerance);
    }

    void updateTimeData(const Plato::OrdinalType aUpdatedTimeStepIndex)
    {
        if (aUpdatedTimeStepIndex < static_cast<Plato::OrdinalType>(0))
        {
            THROWERR("Provided time step index is less than 0.")
        }
        mCurrentTimeStepIndex = aUpdatedTimeStepIndex;
        mCurrentTime = mCurrentTimeStepSize * static_cast<Plato::Scalar>(mCurrentTimeStepIndex + 1);
        mCurrentTime = std::min(mCurrentTime, mEndTime);
        mCurrentTime = std::max(mCurrentTime, mStartTime);
    }

    void increaseNumTimeSteps()
    {
        mNumTimeSteps = static_cast<Plato::OrdinalType>(mNumTimeSteps * mTimeStepExpansionMultiplier);
        if(mNumTimeSteps >= mMaxNumTimeSteps)
        {
            mNumTimeSteps = mMaxNumTimeSteps;
            mMaxNumTimeStepsReached = true;
        }
        mCurrentTimeStepSize  = mEndTime / static_cast<Plato::Scalar>(mNumTimeSteps);
        mCurrentTime          = mStartTime;
        mCurrentTimeStepIndex = static_cast<Plato::OrdinalType>(0);
    }

} // TimeData

} //namespace Plato