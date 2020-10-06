#pragma once

#include "SpatialModel.hpp"
#include "PlatoStaticsTypes.hpp"

namespace Plato
{

/******************************************************************************/
template<typename EvaluationType>
class AbstractLocalVectorFunctionInc
/******************************************************************************/
{
protected:
    const Plato::SpatialDomain     & mSpatialDomain; /*!< Plato spatial model */
          Plato::DataMap           & mDataMap;
          std::vector<std::string>   mDofNames;

public:
    /******************************************************************************/
    explicit 
    AbstractLocalVectorFunctionInc(
        const Plato::SpatialDomain     & aSpatialDomain,
              Plato::DataMap           & aDataMap,
              std::vector<std::string>   aStateNames) :
    /******************************************************************************/
        mSpatialDomain (aSpatialDomain),
        mDataMap       (aDataMap),
        mDofNames      (aStateNames)
    {
    }
    /******************************************************************************/
    virtual ~AbstractLocalVectorFunctionInc()
    /******************************************************************************/
    {
    }

    /****************************************************************************//**
    * \brief Return reference to Omega_h mesh data base 
    ********************************************************************************/
    decltype(mSpatialDomain.Mesh) getMesh() const
    {
        return (mSpatialDomain.Mesh);
    }

    /****************************************************************************//**
    * \brief Return reference to Omega_h mesh sets 
    ********************************************************************************/
    decltype(mSpatialDomain.MeshSets) getMeshSets() const
    {
        return (mSpatialDomain.MeshSets);
    }

    /****************************************************************************//**
    * \brief Return reference to state index map
    ********************************************************************************/
    decltype(mDofNames) getDofNames() const
    {
        return (mDofNames);
    }


    /****************************************************************************//**
    * \brief Evaluate the local residual equations
    ********************************************************************************/
    virtual void
    evaluate(
        const Plato::ScalarMultiVectorT< typename EvaluationType::StateScalarType           > & aGlobalState,
        const Plato::ScalarMultiVectorT< typename EvaluationType::PrevStateScalarType       > & aGlobalStatePrev,
        const Plato::ScalarMultiVectorT< typename EvaluationType::LocalStateScalarType      > & aLocalState,
        const Plato::ScalarMultiVectorT< typename EvaluationType::PrevLocalStateScalarType  > & aLocalStatePrev,
        const Plato::ScalarMultiVectorT< typename EvaluationType::ControlScalarType         > & aControl,
        const Plato::ScalarArray3DT    < typename EvaluationType::ConfigScalarType          > & aConfig,
        const Plato::ScalarMultiVectorT< typename EvaluationType::ResultScalarType          > & aResult,
              Plato::Scalar aTimeStep = 0.0) const = 0;

    /****************************************************************************//**
    * \brief Update the local state variables
    ********************************************************************************/
    virtual void
    updateLocalState(
        const Plato::ScalarMultiVector & aGlobalState,
        const Plato::ScalarMultiVector & aGlobalStatePrev,
        const Plato::ScalarMultiVector & aLocalState,
        const Plato::ScalarMultiVector & aLocalStatePrev,
        const Plato::ScalarMultiVector & aControl,
        const Plato::ScalarArray3D     & aConfig,
              Plato::Scalar              aTimeStep = 0.0) const = 0;

    /******************************************************************************//**
     * \brief Update physics-based data within a frequency of optimization iterations
     * \param [in] aGlobalState global state variables
     * \param [in] aLocalState  local state variables
     * \param [in] aControl     control variables, e.g. design variables
     * \param [in] aTimeStep    pseudo time step
    **********************************************************************************/
    virtual void
    updateProblem(
        const Plato::ScalarMultiVector & aGlobalState,
        const Plato::ScalarMultiVector & aLocalState,
        const Plato::ScalarVector      & aControl,
              Plato::Scalar              aTimeStep = 0.0)
    { return; }
};

} // namespace Plato
