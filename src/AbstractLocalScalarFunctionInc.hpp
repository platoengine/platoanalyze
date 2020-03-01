/*
 * AbstractLocalScalarFunctionInc.hpp
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
 * \brief Abstract scalar function interface for Partial Differential Equations
 *   (PDEs) with path dependent global and local states.
 *
 * \tparam EvaluationType determines the automatic differentiation type used to
 *   evaluate the scalar function (e.g. Value, GradientZ, GradientX, etc.)
 *
*******************************************************************************/
template<typename EvaluationType>
class AbstractLocalScalarFunctionInc
{
// Protected member data
protected:
    Omega_h::Mesh &mMesh;            /*!< mesh database */
    Plato::DataMap &mDataMap;        /*!< output database */
    Omega_h::MeshSets &mMeshSets;    /*!< sideset database */
    const std::string mFunctionName; /*!< my scalar function name */

// public access functions
public:
    /***************************************************************************//**
     * \brief Constructor
     * \param [in] aMesh     mesh metadata
     * \param [in] aMeshSets mesh side-sets metadata
     * \param [in] aDataMap  output data map
     * \param [in] aName     scalar function name, e.g. type
    *******************************************************************************/
    explicit AbstractLocalScalarFunctionInc(Omega_h::Mesh &aMesh,
                                            Omega_h::MeshSets &aMeshSets,
                                            Plato::DataMap &aDataMap,
                                            const std::string & aName) :
        mMesh(aMesh),
        mDataMap(aDataMap),
        mMeshSets(aMeshSets),
        mFunctionName(aName)
    { return; }

    /***************************************************************************//**
     * \brief Destructor
    *******************************************************************************/
    virtual ~AbstractLocalScalarFunctionInc() { return; }

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

    /******************************************************************************//**
     * \brief Return abstract scalar function name
     * \return name
    **********************************************************************************/
    const decltype(mFunctionName)& getName() const
    {
        return (mFunctionName);
    }

    /***************************************************************************//**
     *
     * \brief Evaluate the scalar function with local path-dependent states.
     *
     * \param [in]     aCurrentGlobalState  current global state ( i.e. state at time step n (\f$ t^{n} \f$) )
     * \param [in]     aPreviousGlobalState previous global state ( i.e. state at time step n-1 (\f$ t^{n-1} \f$) )
     * \param [in]     aCurrentLocalState   current local state ( i.e. state at time step n (\f$ t^{n} \f$) )
     * \param [in]     aPreviousLocalState  previous local state ( i.e. state at time step n-1 (\f$ t^{n-1} \f$) )
     * \param [in]     aControls            current set of design variables
     * \param [in]     aConfig              configuration variables, i.e. cell node coordinates
     * \param [in/out] aResult              scalar function value per cell
     * \param [in]     aTimeStep            current time step (i.e. \f$ \Delta{t}^{n} \f$), default = 0.0
     *
    *******************************************************************************/
    virtual void
    evaluate(const Plato::ScalarMultiVectorT<typename EvaluationType::StateScalarType> &aCurrentGlobalState,
             const Plato::ScalarMultiVectorT<typename EvaluationType::PrevStateScalarType> &aPreviousGlobalState,
             const Plato::ScalarMultiVectorT<typename EvaluationType::LocalStateScalarType> &aCurrentLocalState,
             const Plato::ScalarMultiVectorT<typename EvaluationType::PrevLocalStateScalarType> &aPreviousLocalState,
             const Plato::ScalarMultiVectorT<typename EvaluationType::ControlScalarType> &aControls,
             const Plato::ScalarArray3DT<typename EvaluationType::ConfigScalarType> &aConfig,
             const Plato::ScalarVectorT<typename EvaluationType::ResultScalarType> &aResult,
             Plato::Scalar aTimeStep = 0.0) = 0;

    /******************************************************************************//**
     * \brief Update physics-based data in between optimization iterations
     * \param [in] aGlobalState global state variables
     * \param [in] aLocalState  local state variables
     * \param [in] aControl     control variables, e.g. design variables
    **********************************************************************************/
    virtual void updateProblem(const Plato::ScalarMultiVector & aGlobalState,
                               const Plato::ScalarMultiVector & aLocalState,
                               const Plato::ScalarVector & aControl)
    { return; }
};
// class AbstractLocalScalarFunctionInc

}
// namespace Plato
