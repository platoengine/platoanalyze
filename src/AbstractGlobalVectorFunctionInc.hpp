/*
 * AbstractGlobalVectorFunctionInc.hpp
 *
 *  Created on: Feb 29, 2020
 */

#pragma once

#include <Omega_h_mesh.hpp>
#include <Omega_h_assoc.hpp>

#include "PlatoStaticsTypes.hpp"

namespace Plato
{

/***************************************************************************//**
 *
 * \brief Abstract vector function interface for Variational Multi-Scale (VMS)
 *   Partial Differential Equations (PDEs) with history dependent states
 *
 * \tparam EvaluationType evaluation type use to determine automatic differentiation
 *   type for the vector function (e.g. Residual, Jacobian, GradientZ, GradientU, etc.)
 *
*******************************************************************************/
template<typename EvaluationType>
class AbstractGlobalVectorFunctionInc
{
// Protected member data
protected:
    Omega_h::Mesh &mMesh; /*!< mesh database */
    Plato::DataMap &mDataMap; /*!< output database */
    Omega_h::MeshSets &mMeshSets; /*!< sideset database */

// Public access functions
public:
    /***************************************************************************//**
     * \brief Constructor
     * \param [in]  aMesh mesh metadata
     * \param [in]  aMeshSets mesh side-sets metadata
     * \param [in]  aDataMap output data map
    *******************************************************************************/
    explicit AbstractGlobalVectorFunctionInc(Omega_h::Mesh &aMesh,
                                             Omega_h::MeshSets &aMeshSets,
                                             Plato::DataMap &aDataMap) :
        mMesh(aMesh),
        mDataMap(aDataMap),
        mMeshSets(aMeshSets)
    {
    }

    /***************************************************************************//**
     * \brief Destructor
    *******************************************************************************/
    virtual ~AbstractGlobalVectorFunctionInc()
    {
    }

    /***************************************************************************//**
     * \brief Return reference to Omega_h mesh data base
     * \return mesh metadata
    *******************************************************************************/
    decltype(mMesh) getMesh() const
    {
        return (mMesh);
    }

    /***************************************************************************//**
     * \brief Return reference to Omega_h mesh sets
     * \return mesh side sets metadata
    *******************************************************************************/
    decltype(mMeshSets) getMeshSets() const
    {
        return (mMeshSets);
    }

    /***************************************************************************//**
     *
     * \brief Evaluate stabilized residual
     *
     * \param [in]     aGlobalState     current global state ( i.e. global state at time step n )
     * \param [in]     aGlobalStatePrev previous global state ( i.e. global state at time step n-1 )
     * \param [in]     aLocalState      current local state ( i.e. local state at time step n )
     * \param [in]     aLocalStatePrev  previous local state ( i.e. local state at time step n-1 )
     * \param [in]     aPressureGrad    current pressure gradient ( i.e. projected pressure gradient at time step n-1 )
     * \param [in]     aControls        set of design variables
     * \param [in]     aConfig          set of configuration variables (cell node coordinates)
     * \param [in/out] aResult          residual evaluation
     * \param [in]     aTimeStep        current time step, default = 0.0
     *
    *******************************************************************************/
    virtual void
    evaluate(const Plato::ScalarMultiVectorT<typename EvaluationType::StateScalarType> &aGlobalState,
             const Plato::ScalarMultiVectorT<typename EvaluationType::PrevStateScalarType> &aGlobalStatePrev,
             const Plato::ScalarMultiVectorT<typename EvaluationType::LocalStateScalarType> &aLocalState,
             const Plato::ScalarMultiVectorT<typename EvaluationType::PrevLocalStateScalarType> &aLocalStatePrev,
             const Plato::ScalarMultiVectorT<typename EvaluationType::NodeStateScalarType> &aPressureGrad,
             const Plato::ScalarMultiVectorT<typename EvaluationType::ControlScalarType> &aControls,
             const Plato::ScalarArray3DT<typename EvaluationType::ConfigScalarType> &aConfig,
             const Plato::ScalarMultiVectorT<typename EvaluationType::ResultScalarType> &aResult,
             Plato::Scalar aTimeStep = 0.0) = 0;

    /******************************************************************************//**
     * \brief Update physics-based data within a frequency of optimization iterations
     * \param [in] aGlobalState global state variables
     * \param [in] aLocalState  local state variables
     * \param [in] aControl     control variables, e.g. design variables
     * \param [in] aTimeStep    pseudo time step
    **********************************************************************************/
    virtual void
    updateProblem(const Plato::ScalarMultiVector & aGlobalState,
                  const Plato::ScalarMultiVector & aLocalState,
                  const Plato::ScalarVector & aControl,
                  Plato::Scalar aTimeStep = 0.0)
    { return; }
};
// class AbstractGlobalVectorFunctionInc

}
// namespace Plato
