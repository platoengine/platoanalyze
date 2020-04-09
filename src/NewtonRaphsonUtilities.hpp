/*
 * NewtonRaphsonUtilities.hpp
 *
 *  Created on: Mar 2, 2020
 */

#pragma once

#include <locale>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "AnalyzeMacros.hpp"
#include "PlatoMathHelpers.hpp"
#include "PlatoStaticsTypes.hpp"

namespace Plato
{

/******************************************************************************//**
 * \brief C++ structure holding enum types used by the Newton-Raphson solver.
**********************************************************************************/
struct NewtonRaphson
{
    enum stop_t
    {
        DID_NOT_CONVERGE = 0,
        MAX_NUMBER_ITERATIONS = 1,
        NORM_MEASURE_TOLERANCE = 2,
        CURRENT_NORM_TOLERANCE = 3,
        NaN_NORM_VALUE = 4,
    };

    enum measure_t
    {
        DISPLACEMENT_NORM = 0,
        ABSOLUTE_RESIDUAL_NORM = 1,
        RELATIVE_RESIDUAL_NORM = 2,
    };
};

/***************************************************************************//**
 * \brief C++ structure used to solve path-dependent forward problems. Basically,
 * at a given time snapshot, this C++ structures provide the most recent set
 * of local and global states.
*******************************************************************************/
struct CurrentStates
{
    Plato::OrdinalType mCurrentStepIndex;      /*!< current time step index */

    Plato::ScalarVector mDeltaGlobalState;     /*!< global state increment */
    Plato::ScalarVector mCurrentLocalState;    /*!< current local state */
    Plato::ScalarVector mPreviousLocalState;   /*!< previous local state */
    Plato::ScalarVector mCurrentGlobalState;   /*!< current global state */
    Plato::ScalarVector mPreviousGlobalState;  /*!< previous global state */
    Plato::ScalarVector mProjectedPressGrad;   /*!< current projected pressure gradient */
};
// struct CurrentStates

/******************************************************************************//**
 * \brief C++ structure holding the output diagnostics for the Newton-Raphson solver.
**********************************************************************************/
struct NewtonRaphsonOutputData
{
    bool mWriteOutput;              /*!< flag: true = write output; false = do not write output */
    Plato::Scalar mCurrentNorm;     /*!< current norm */
    Plato::Scalar mNormMeasure;    /*!< relative norm */
    Plato::Scalar mReferenceNorm;   /*!< reference norm */

    Plato::OrdinalType mCurrentIteration;             /*!< current Newton-Raphson solver iteration */
    Plato::NewtonRaphson::stop_t mStopingCriterion;   /*!< stopping criterion */
    Plato::NewtonRaphson::measure_t mStoppingMeasure; /*!< stopping criterion measure */

    /******************************************************************************//**
     * \brief Constructor
    **********************************************************************************/
    NewtonRaphsonOutputData() :
        mWriteOutput(true),
        mCurrentNorm(std::numeric_limits<Plato::Scalar>::max()),
        mReferenceNorm(std::numeric_limits<Plato::Scalar>::max()),
        mNormMeasure(std::numeric_limits<Plato::Scalar>::max()),
        mCurrentIteration(0),
        mStopingCriterion(Plato::NewtonRaphson::DID_NOT_CONVERGE),
        mStoppingMeasure(Plato::NewtonRaphson::ABSOLUTE_RESIDUAL_NORM)
    {}
};
// struct NewtonRaphsonOutputData


/******************************************************************************//**
 * \brief Return Newton-Raphson solver's stopping criterion
 * \param [in] aInput string with stopping criterion
 * \return stopping criterion enum
**********************************************************************************/
inline Plato::NewtonRaphson::measure_t newton_raphson_stopping_criterion(const std::string& aInput)
{
    // convert string to upper case
    std::string tCopy = aInput;
    std::for_each(tCopy.begin(), tCopy.end(), [](char & aChar)
    {
        aChar = std::toupper(aChar);
    });

    if(tCopy.compare("ABSOLUTE RESIDUAL NORM") == 0)
    {
        return Plato::NewtonRaphson::ABSOLUTE_RESIDUAL_NORM;
    }
    else if(tCopy.compare("RELATIVE RESIDUAL NORM") == 0)
    {
        return Plato::NewtonRaphson::RELATIVE_RESIDUAL_NORM;
    }
    else
    {
        std::ostringstream tMsg;
        tMsg << "Newton-Raphson Stopping Criterion '" << aInput.c_str() << "' is not defined as a valid stopping criterion. "
                << "Valid Options: 'ABSOLUTE RESIDUAL NORM' and 'RELATIVE RESIDUAL NORM'";
        THROWERR(tMsg.str().c_str())
    }
}
// function newton_raphson_stopping_criterion

/******************************************************************************//**
 * \brief Writes a brief sentence explaining why the Newton-Raphson solver stopped.
 * \param [in]     aOutputData Newton-Raphson solver output data
 * \param [in,out] aOutputFile Newton-Raphson solver output file
**********************************************************************************/
inline void print_newton_raphson_stop_criterion(const Plato::NewtonRaphsonOutputData & aOutputData, std::ofstream &aOutputFile)
{
    if(aOutputData.mWriteOutput == false)
    {
        return;
    }

    if(aOutputFile.is_open() == false)
    {
        THROWERR("Newton-Raphson solver diagnostic file is closed.")
    }

    switch(aOutputData.mStopingCriterion)
    {
        case Plato::NewtonRaphson::MAX_NUMBER_ITERATIONS:
        {
            aOutputFile << "\n\n****** Newton-Raphson solver stopping due to exceeding maximum number of iterations. ******\n\n";
            break;
        }
        case Plato::NewtonRaphson::NORM_MEASURE_TOLERANCE:
        {
            aOutputFile << "\n\n******  Newton-Raphson algorithm stopping due to norm measure being met. ******\n\n";
            break;
        }
        case Plato::NewtonRaphson::CURRENT_NORM_TOLERANCE:
        {
            aOutputFile << "\n\n******  Newton-Raphson algorithm stopping due to current norm tolerance being met. ******\n\n";
            break;
        }
        case Plato::NewtonRaphson::NaN_NORM_VALUE:
        {
            aOutputFile << "\n\n******  MAJOR FAILURE: Newton-Raphson algorithm stopping due to NaN norm value detected. ******\n\n";
            break;
        }
        case Plato::NewtonRaphson::DID_NOT_CONVERGE:
        {
            aOutputFile << "\n\n****** Newton-Raphson algorithm did not converge. ******\n\n";
            break;
        }
        default:
        {
            aOutputFile << "\n\n****** ERROR: Optimization algorithm stopping due to undefined behavior. ******\n\n";
            break;
        }
    }
}
// function print_newton_raphson_stop_criterion

/******************************************************************************//**
 * \brief Writes the Newton-Raphson solver diagnostics for the current iteration.
 * \param [in]     aOutputData Newton-Raphson solver output data
 * \param [in,out] aOutputFile Newton-Raphson solver output file
**********************************************************************************/
inline void print_newton_raphson_diagnostics(const Plato::NewtonRaphsonOutputData & aOutputData, std::ofstream &aOutputFile)
{
    if(aOutputData.mWriteOutput == false)
    {
        return;
    }

    if(aOutputFile.is_open() == false)
    {
        THROWERR("Newton-Raphson solver diagnostic file is closed.")
    }

    aOutputFile << std::scientific << std::setprecision(6) << aOutputData.mCurrentIteration << std::setw(20)
        << aOutputData.mCurrentNorm << std::setw(20) << aOutputData.mNormMeasure << "\n" << std::flush;
}
// function print_newton_raphson_diagnostics

/******************************************************************************//**
 * \brief Writes the header for the Newton-Raphson solver diagnostics output file.
 * \param [in]     aOutputData Newton-Raphson solver output data
 * \param [in,out] aOutputFile Newton-Raphson solver output file
**********************************************************************************/
inline void print_newton_raphson_diagnostics_header(const Plato::NewtonRaphsonOutputData &aOutputData, std::ofstream &aOutputFile)
{
    if(aOutputData.mWriteOutput == false)
    {
        return;
    }

    if(aOutputFile.is_open() == false)
    {
        THROWERR("Newton-Raphson solver diagnostic file is closed.")
    }

    aOutputFile << std::scientific << std::setprecision(6) << std::right << "Iter" << std::setw(13)
        << "Norm" << std::setw(22) << "Relative" "\n" << std::flush;
}
// function print_newton_raphson_diagnostics_header

/******************************************************************************//**
 * \brief Computes relative residual norm criterion, i.e.
 *
 * \f$ \mbox{relative error} = \frac{\Vert \hat{\alpha} - \alpha \Vert}{\Vert \hat{\alpha} \Vert} \f$,
 *
 * where \f$ \hat{\alpha} \f$ is an approximation of the true answer \f$ \alpha \f$.
 *
 * \param [in]     aResidual   current residual vector
 * \param [in,out] aOutputData C++ structure with Newton-Raphson solver output data
**********************************************************************************/
inline void compute_relative_residual_norm_error(const Plato::ScalarVector & aResidual, Plato::NewtonRaphsonOutputData & aOutputData)
{
    if(std::isfinite(aOutputData.mCurrentNorm) == false)
    {
        THROWERR("Relative Error Calculation: Current norm value is not a finite number.")
    }
    if(std::isfinite(aOutputData.mReferenceNorm) == false)
    {
        THROWERR("Relative Error Calculation: Reference norm value is not a finite number.")
    }

    if(aOutputData.mCurrentIteration == static_cast<Plato::OrdinalType>(0))
    {
        aOutputData.mReferenceNorm = Plato::norm(aResidual);
        aOutputData.mCurrentNorm = aOutputData.mReferenceNorm;
    }
    else
    {
        aOutputData.mCurrentNorm = Plato::norm(aResidual);
        aOutputData.mNormMeasure = std::abs(aOutputData.mReferenceNorm - aOutputData.mCurrentNorm) / std::abs(aOutputData.mCurrentNorm);
        aOutputData.mReferenceNorm = aOutputData.mCurrentNorm;
    }
}
// function compute_relative_residual_norm_error

/******************************************************************************//**
 * \brief Computes absolute residual norm criterion.
 *
 * \f$ \mbox{absolute error} = \Vert \hat{\alpha} - \alpha \Vert \f$,
 *
 * where \f$ \hat{\alpha} \f$ is an approximation of the true answer \f$ \alpha \f$.
 *
 * \param [in]     aResidual   current residual vector
 * \param [in,out] aOutputData C++ structure with Newton-Raphson solver output data
**********************************************************************************/
inline void compute_absolute_residual_norm_error(const Plato::ScalarVector & aResidual, Plato::NewtonRaphsonOutputData & aOutputData)
{
    if(std::isfinite(aOutputData.mCurrentNorm) == false)
    {
        THROWERR("Relative Error Calculation: Current norm value is not a finite number.")
    }
    if(std::isfinite(aOutputData.mReferenceNorm) == false)
    {
        THROWERR("Relative Error Calculation: Reference norm value is not a finite number.")
    }

    if(aOutputData.mCurrentIteration == static_cast<Plato::OrdinalType>(0))
    {
        aOutputData.mReferenceNorm = Plato::norm(aResidual);
        aOutputData.mCurrentNorm = aOutputData.mReferenceNorm;
    }
    else
    {
        aOutputData.mCurrentNorm = Plato::norm(aResidual);
        aOutputData.mNormMeasure = std::abs(aOutputData.mReferenceNorm - aOutputData.mCurrentNorm);
        aOutputData.mReferenceNorm = aOutputData.mCurrentNorm;
    }
}
// function compute_relative_residual_norm_error

}
// namespace Plato
