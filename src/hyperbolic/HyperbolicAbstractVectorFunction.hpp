#ifndef ABSTRACT_VECTOR_FUNCTION_HYPERBOLIC_HPP
#define ABSTRACT_VECTOR_FUNCTION_HYPERBOLIC_HPP

#include <Omega_h_mesh.hpp>
#include <Omega_h_assoc.hpp>

#include "Solutions.hpp"

namespace Plato
{

namespace Hyperbolic
{

/******************************************************************************/
template<typename EvaluationType>
class AbstractVectorFunction
/******************************************************************************/
{
protected:
    const Plato::SpatialDomain & mSpatialDomain;

    Plato::DataMap& mDataMap;
    std::vector<std::string> mDofNames;
    std::vector<std::string> mDofDotNames;

public:
    /******************************************************************************/
    explicit 
    AbstractVectorFunction(
        const Plato::SpatialDomain     & aSpatialDomain,
              Plato::DataMap           & aDataMap,
              std::vector<std::string>   aStateNames,
              std::vector<std::string>   aStateDotNames
    ) :
    /******************************************************************************/
        mSpatialDomain (aSpatialDomain),
        mDataMap       (aDataMap),
        mDofNames      (aStateNames),
        mDofDotNames   (aStateDotNames)
    {
    }
    /******************************************************************************/
    virtual ~AbstractVectorFunction()
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
    * \brief Return reference to state dot index map
    ********************************************************************************/
    decltype(mDofDotNames) getDofDotNames() const
    {
        return (mDofDotNames);
    }

    /**************************************************************************//**
    * \brief Call the output state function in the residual
    * \param [in] aSolutions State solutions database
    * \return output solutions database
    ******************************************************************************/
    virtual Plato::Solutions 
    getSolutionStateOutputData(const Plato::Solutions &aSolutions) const = 0;

    /******************************************************************************/
    virtual void
    evaluate(
        const Plato::ScalarMultiVectorT< typename EvaluationType::StateScalarType       > & aState,
        const Plato::ScalarMultiVectorT< typename EvaluationType::StateDotScalarType    > & aStateDot,
        const Plato::ScalarMultiVectorT< typename EvaluationType::StateDotDotScalarType > & aStateDotDot,
        const Plato::ScalarMultiVectorT< typename EvaluationType::ControlScalarType     > & aControl,
        const Plato::ScalarArray3DT    < typename EvaluationType::ConfigScalarType      > & aConfig,
              Plato::ScalarMultiVectorT< typename EvaluationType::ResultScalarType      > & aResult,
              Plato::Scalar aTimeStep = 0.0, 
              Plato::Scalar aCurrentTime = 0.0) const = 0;
    /******************************************************************************/

    /******************************************************************************/
    virtual void
    evaluate_boundary(
        const Plato::SpatialModel                                                         & aSpatialModel,
        const Plato::ScalarMultiVectorT< typename EvaluationType::StateScalarType       > & aState,
        const Plato::ScalarMultiVectorT< typename EvaluationType::StateDotScalarType    > & aStateDot,
        const Plato::ScalarMultiVectorT< typename EvaluationType::StateDotDotScalarType > & aStateDotDot,
        const Plato::ScalarMultiVectorT< typename EvaluationType::ControlScalarType     > & aControl,
        const Plato::ScalarArray3DT    < typename EvaluationType::ConfigScalarType      > & aConfig,
              Plato::ScalarMultiVectorT< typename EvaluationType::ResultScalarType      > & aResult,
              Plato::Scalar aTimeStep = 0.0, 
              Plato::Scalar aCurrentTime = 0.0) const = 0;
    /******************************************************************************/
};

} // namespace Hyperbolic

} // namespace Plato

#endif
