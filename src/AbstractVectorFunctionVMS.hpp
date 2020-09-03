#ifndef ABSTRACT_VECTOR_FUNCTION_VMS_HPP
#define ABSTRACT_VECTOR_FUNCTION_VMS_HPP

#include "PlatoStaticsTypes.hpp"
#include "SpatialModel.hpp"

namespace Plato
{

/******************************************************************************//**
 * \brief Abstract vector function (i.e. PDE) interface for Variational Multi-Scale (VMS)
 * \tparam EvaluationType evaluation type use to determine automatic differentiation
 *   type for scalar function (e.g. Residual, Jacobian, GradientZ, etc.)
 **********************************************************************************/
template<typename EvaluationType>
class AbstractVectorFunctionVMS
{
protected:
    const Plato::SpatialDomain & mSpatialDomain; /*!< Plato Analyze spatial model */
          Plato::DataMap       & mDataMap;       /*!< Plato Analyze database */

public:
    /******************************************************************************//**
     * \brief Constructor
     * \param [in] aSpatialDomain Plato Analyze spatial model
     * \param [in] aDataMap Plato Analyze database
    **********************************************************************************/
    explicit
    AbstractVectorFunctionVMS(
        const Plato::SpatialDomain & aSpatialDomain,
              Plato::DataMap& aDataMap
    ) :
        mSpatialDomain (aSpatialDomain),
        mDataMap       (aDataMap)
    {
    }

    /******************************************************************************//**
     * \brief Destructor
    **********************************************************************************/
    virtual ~AbstractVectorFunctionVMS()
    {
    }

    /****************************************************************************//**
    * \brief Return reference to Omega_h mesh database
    * \return volume mesh database
    ********************************************************************************/
    decltype(mSpatialDomain.Mesh) getMesh() const
    {
        return (mSpatialDomain.Mesh);
    }

    /****************************************************************************//**
    * \brief Return reference to Omega_h mesh sets
    * \return surface mesh database
    ********************************************************************************/
    decltype(mSpatialDomain.MeshSets) getMeshSets() const
    {
        return (mSpatialDomain.MeshSets);
    }

    /******************************************************************************//**
     * \brief Evaluate vector function
     * \param [in] aState 2D array with state variables (C,DOF)
     * \param [in] aNodeState 2D array with state variables (C,D*N)
     * \param [in] aControl 2D array with control variables (C,N)
     * \param [in] aConfig 3D array with control variables (C,N,D)
     * \param [in] aResult 1D array with control variables (C,DOF)
     * \param [in] aTimeStep current time step
     * Nomenclature: C = number of cells, DOF = number of degrees of freedom per cell
     * N = number of nodes per cell, D = spatial dimensions
    **********************************************************************************/
    virtual void
    evaluate(
        const Plato::ScalarMultiVectorT <typename EvaluationType::StateScalarType>     & aState,
        const Plato::ScalarMultiVectorT <typename EvaluationType::NodeStateScalarType> & aNodeState,
        const Plato::ScalarMultiVectorT <typename EvaluationType::ControlScalarType>   & aControl,
        const Plato::ScalarArray3DT     <typename EvaluationType::ConfigScalarType>    & aConfig,
              Plato::ScalarMultiVectorT <typename EvaluationType::ResultScalarType>    & aResult,
              Plato::Scalar aTimeStep = 0.0) const = 0;

    /******************************************************************************//**
     * \brief Evaluate vector function
     * \param [in] aState 2D array with state variables (C,DOF)
     * \param [in] aNodeState 2D array with state variables (C,D*N)
     * \param [in] aControl 2D array with control variables (C,N)
     * \param [in] aConfig 3D array with control variables (C,N,D)
     * \param [in] aResult 1D array with control variables (C,DOF)
     * \param [in] aTimeStep current time step
     * Nomenclature: C = number of cells, DOF = number of degrees of freedom per cell
     * N = number of nodes per cell, D = spatial dimensions
    **********************************************************************************/
    virtual void
    evaluate_boundary(
        const Plato::SpatialModel                                                      & aModel,
        const Plato::ScalarMultiVectorT <typename EvaluationType::StateScalarType>     & aState,
        const Plato::ScalarMultiVectorT <typename EvaluationType::NodeStateScalarType> & aNodeState,
        const Plato::ScalarMultiVectorT <typename EvaluationType::ControlScalarType>   & aControl,
        const Plato::ScalarArray3DT     <typename EvaluationType::ConfigScalarType>    & aConfig,
              Plato::ScalarMultiVectorT <typename EvaluationType::ResultScalarType>    & aResult,
              Plato::Scalar aTimeStep = 0.0) const {}

    /******************************************************************************//**
     * \brief Update physics-based data within a frequency of optimization iterations
     * \param [in] aGlobalState global state variables
     * \param [in] aControl     control variables, e.g. design variables
     * \param [in] aTimeStep    pseudo time step
    **********************************************************************************/
    virtual void
    updateProblem(
        const Plato::ScalarMultiVector & aState,
        const Plato::ScalarVector      & aControl,
              Plato::Scalar              aTimeStep = 0.0)
    { return; }
};
// class AbstractVectorFunctionVMS

} // namespace Plato

#endif
