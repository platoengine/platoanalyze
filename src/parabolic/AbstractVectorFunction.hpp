#ifndef ABSTRACT_VECTOR_FUNCTION_PARABOLIC_HPP
#define ABSTRACT_VECTOR_FUNCTION_PARABOLIC_HPP

#include "SpatialModel.hpp"
#include "PlatoStaticsTypes.hpp"

namespace Plato
{

namespace Parabolic
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

public:
    /******************************************************************************/
    explicit 
    AbstractVectorFunction(
        const Plato::SpatialDomain     & aSpatialDomain,
              Plato::DataMap           & aDataMap,
              std::vector<std::string>   aStateNames
    ) :
    /******************************************************************************/
        mSpatialDomain (aSpatialDomain),
        mDataMap       (aDataMap),
        mDofNames      (aStateNames)
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


    /******************************************************************************/
    virtual void
    evaluate(
        const Plato::ScalarMultiVectorT< typename EvaluationType::StateScalarType    > & aState,
        const Plato::ScalarMultiVectorT< typename EvaluationType::StateDotScalarType > & aStateDot,
        const Plato::ScalarMultiVectorT< typename EvaluationType::ControlScalarType  > & aControl,
        const Plato::ScalarArray3DT    < typename EvaluationType::ConfigScalarType   > & aConfig,
              Plato::ScalarMultiVectorT< typename EvaluationType::ResultScalarType   > & aResult,
              Plato::Scalar aTimeStep = 0.0) const = 0;
    /******************************************************************************/

    /******************************************************************************/
    virtual void
    evaluate_boundary(
        const Plato::SpatialModel                                                      & aModel,
        const Plato::ScalarMultiVectorT< typename EvaluationType::StateScalarType    > & aState,
        const Plato::ScalarMultiVectorT< typename EvaluationType::StateDotScalarType > & aStateDot,
        const Plato::ScalarMultiVectorT< typename EvaluationType::ControlScalarType  > & aControl,
        const Plato::ScalarArray3DT    < typename EvaluationType::ConfigScalarType   > & aConfig,
              Plato::ScalarMultiVectorT< typename EvaluationType::ResultScalarType   > & aResult,
              Plato::Scalar aTimeStep = 0.0) const = 0;
    /******************************************************************************/
};

} // namespace Parabolic

} // namespace Plato

#endif
