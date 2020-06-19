#ifndef ABSTRACT_VECTOR_FUNCTION_PARABOLIC_HPP
#define ABSTRACT_VECTOR_FUNCTION_PARABOLIC_HPP

#include <Omega_h_mesh.hpp>
#include <Omega_h_assoc.hpp>

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
    Omega_h::Mesh& mMesh;
    Plato::DataMap& mDataMap;
    Omega_h::MeshSets& mMeshSets;
    std::vector<std::string> mDofNames;

public:
    /******************************************************************************/
    explicit 
    AbstractVectorFunction(
        Omega_h::Mesh& aMesh, 
        Omega_h::MeshSets& aMeshSets,
        Plato::DataMap& aDataMap,
        std::vector<std::string> aStateNames) :
    /******************************************************************************/
            mMesh(aMesh),
            mDataMap(aDataMap),
            mMeshSets(aMeshSets),
            mDofNames(aStateNames)
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
    decltype(mMesh) getMesh() const
    {
        return (mMesh);
    }

    /****************************************************************************//**
    * \brief Return reference to Omega_h mesh sets 
    ********************************************************************************/
    decltype(mMeshSets) getMeshSets() const
    {
        return (mMeshSets);
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
    evaluate(const Plato::ScalarMultiVectorT< typename EvaluationType::StateScalarType    > & aState,
             const Plato::ScalarMultiVectorT< typename EvaluationType::StateDotScalarType > & aStateDot,
             const Plato::ScalarMultiVectorT< typename EvaluationType::ControlScalarType  > & aControl,
             const Plato::ScalarArray3DT    < typename EvaluationType::ConfigScalarType   > & aConfig,
             Plato::ScalarMultiVectorT<typename EvaluationType::ResultScalarType> & aResult,
             Plato::Scalar aTimeStep = 0.0) const = 0;
    /******************************************************************************/
};

} // namespace Parabolic

} // namespace Plato

#endif