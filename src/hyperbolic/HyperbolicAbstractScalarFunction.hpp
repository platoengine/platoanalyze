#ifndef HYPERBOLIC_ABSTRACT_SCALAR_FUNCTION
#define HYPERBOLIC_ABSTRACT_SCALAR_FUNCTION

#include "SpatialModel.hpp"
#include "PlatoStaticsTypes.hpp"

namespace Plato
{

namespace Hyperbolic
{

/******************************************************************************//**
 * \brief Abstract scalar function (i.e. criterion) interface
 * @tparam EvaluationType evaluation type use to determine automatic differentiation
 *   type for scalar function (e.g. Residual, Jacobian, GradientZ, etc.)
 **********************************************************************************/
template<typename EvaluationType>
class AbstractScalarFunction
{
protected:
    const Plato::SpatialDomain & mSpatialDomain; /*!< Plato spatial model */
          Plato::DataMap       & mDataMap;       /*!< Plato Analyze database */
    const std::string            mFunctionName;  /*!< name of scalar function */

public:
    /******************************************************************************//**
     * \brief Constructor
     * \param [in] aSpatialDomain Plato Analyze spatial domain
     * \param [in] aDataMap Plato Analyze database
     * \param [in] aName name of scalar function
    **********************************************************************************/
    AbstractScalarFunction(
        const Plato::SpatialDomain & aSpatialDomain,
              Plato::DataMap       & aDataMap,
        const std::string          & aName
    ) :
        mSpatialDomain (aSpatialDomain),
        mDataMap       (aDataMap),
        mFunctionName  (aName)
    {
    }

    /******************************************************************************//**
     * \brief Destructor
    **********************************************************************************/
    virtual ~AbstractScalarFunction()
    {
    }

    /******************************************************************************//**
     * \brief Evaluate time-dependent scalar function
     * \param [in] aState 2D array with current state variables (C,DOF)
     * \param [in] aStateDot 2D array with state first derivative variables (C,DOF)
     * \param [in] aStateDotDot 2D array with state second derivative variables (C,DOF)
     * \param [in] aControl 2D array with control variables (C,N)
     * \param [in] aConfig 3D array with control variables (C,N,D)
     * \param [in] aResult 1D array with control variables (C)
     * \param [in] aTimeStep current time step
     * Nomenclature: C = number of cells, DOF = number of degrees of freedom per cell
     * N = number of nodes per cell, D = spatial dimensions
    **********************************************************************************/
    virtual void
    evaluate(
        const Plato::ScalarMultiVectorT <typename EvaluationType::StateScalarType>        & aState,
        const Plato::ScalarMultiVectorT <typename EvaluationType::StateDotScalarType>     & aStateDot,
        const Plato::ScalarMultiVectorT <typename EvaluationType::StateDotDotScalarType>  & aStateDotDot,
        const Plato::ScalarMultiVectorT <typename EvaluationType::ControlScalarType>      & aControl,
        const Plato::ScalarArray3DT     <typename EvaluationType::ConfigScalarType>       & aConfig,
              Plato::ScalarVectorT      <typename EvaluationType::ResultScalarType>       & aResult,
              Plato::Scalar aTimeStep = 0.0) const = 0;

    /******************************************************************************//**
     * \brief Post-evaluate time-dependent scalar function after evaluate call
     * \param [in] aInput 1D array with scalar function values (C)
     * \param [in] aScalar scalar multiplier
    **********************************************************************************/
    virtual void postEvaluate(Plato::ScalarVector aInput, Plato::Scalar aScalar)
    { return; }

    /******************************************************************************//**
     * \brief Post-evaluate time-dependent scalar function after evaluate call
     * \param [in] aScalar scalar multiplier
    **********************************************************************************/
    virtual void postEvaluate(Plato::Scalar&)
    { return; }

    /******************************************************************************//**
     * \brief Return name of time-dependent scalar function
     * \return function name
    **********************************************************************************/
    const decltype(mFunctionName)& getName()
    {
        return mFunctionName;
    }
}; // class AbstractScalarFunction

}// namespace Hyperbolic

}// namespace Plato

#endif
