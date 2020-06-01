#pragma once

#include <memory>
#include <cassert>
#include <vector>

#include <Omega_h_mesh.hpp>

#include "PlatoStaticsTypes.hpp"
#include "geometric/WorksetBase.hpp"
#include "geometric/ScalarFunctionBase.hpp"
#include "geometric/ScalarFunctionBaseFactory.hpp"
#include <Teuchos_ParameterList.hpp>

namespace Plato
{

namespace Geometric
{

/******************************************************************************//**
 * @brief Division function class \f$ F(x) = numerator(x) / denominator(x) \f$
 **********************************************************************************/
template<typename PhysicsT>
class DivisionFunction : public Plato::Geometric::ScalarFunctionBase, public Plato::Geometric::WorksetBase<PhysicsT>
{
private:
    using Plato::Geometric::WorksetBase<PhysicsT>::mNumSpatialDims; /*!< number of spatial dimensions */
    using Plato::Geometric::WorksetBase<PhysicsT>::mNumNodes; /*!< total number of nodes in the mesh */

    std::shared_ptr<Plato::Geometric::ScalarFunctionBase> mScalarFunctionBaseNumerator; /*!< numerator function */
    std::shared_ptr<Plato::Geometric::ScalarFunctionBase> mScalarFunctionBaseDenominator; /*!< denominator function */

    Plato::DataMap& mDataMap; /*!< PLATO Engine and Analyze data map */

    std::string mFunctionName; /*!< User defined function name */

	/******************************************************************************//**
     * @brief Initialization of Division Function
     * @param [in] aInputParams input parameters database
    **********************************************************************************/
    void initialize (Omega_h::Mesh& aMesh, 
                     Omega_h::MeshSets& aMeshSets, 
                     Teuchos::ParameterList & aInputParams)
    {
        Plato::Geometric::ScalarFunctionBaseFactory<PhysicsT> tFactory;

        auto tProblemFunctionName = aInputParams.sublist(mFunctionName);

        auto tNumeratorFunctionName = tProblemFunctionName.get<std::string>("Numerator");
        auto tDenominatorFunctionName = tProblemFunctionName.get<std::string>("Denominator");

        mScalarFunctionBaseNumerator = 
             tFactory.create(aMesh, aMeshSets, mDataMap, aInputParams, tNumeratorFunctionName);

        mScalarFunctionBaseDenominator = 
             tFactory.create(aMesh, aMeshSets, mDataMap, aInputParams, tDenominatorFunctionName);
    }

public:
    /******************************************************************************//**
     * @brief Primary division function constructor
     * @param [in] aMesh mesh database
     * @param [in] aMeshSets side sets database
     * @param [in] aDataMap PLATO Engine and Analyze data map
     * @param [in] aInputParams input parameters database
     * @param [in] aName user defined function name
    **********************************************************************************/
    DivisionFunction(Omega_h::Mesh& aMesh,
                Omega_h::MeshSets& aMeshSets,
                Plato::DataMap & aDataMap,
                Teuchos::ParameterList& aInputParams,
                std::string& aName) :
            Plato::Geometric::WorksetBase<PhysicsT>(aMesh),
            mDataMap(aDataMap),
            mFunctionName(aName)
    {
        initialize(aMesh, aMeshSets, aInputParams);
    }

    /******************************************************************************//**
     * @brief Secondary division function constructor, used for unit testing
     * @param [in] aMesh mesh database
     * @param [in] aMeshSets side sets database
    **********************************************************************************/
    DivisionFunction(Omega_h::Mesh& aMesh, Plato::DataMap& aDataMap) :
            Plato::Geometric::WorksetBase<PhysicsT>(aMesh),
            mDataMap(aDataMap),
            mFunctionName("Division Function")
    {
    }

    /******************************************************************************//**
     * @brief Allocate numerator function base using the residual automatic differentiation type
     * @param [in] aInput scalar function
    **********************************************************************************/
    void allocateNumeratorFunction(const std::shared_ptr<Plato::Geometric::ScalarFunctionBase>& aInput)
    {
        mScalarFunctionBaseNumerator = aInput;
    }

    /******************************************************************************//**
     * @brief Allocate denominator function base using the residual automatic differentiation type
     * @param [in] aInput scalar function
    **********************************************************************************/
    void allocateDenominatorFunction(const std::shared_ptr<Plato::Geometric::ScalarFunctionBase>& aInput)
    {
        mScalarFunctionBaseDenominator = aInput;
    }

    /******************************************************************************//**
     * @brief Update physics-based parameters within optimization iterations
     * @param [in] aControl 1D view of control variables
     **********************************************************************************/
    void updateProblem(const Plato::ScalarVector & aControl) const
    {
        mScalarFunctionBaseNumerator->updateProblem(aControl);
        mScalarFunctionBaseDenominator->updateProblem(aControl);
    }

    /******************************************************************************//**
     * @brief Evaluate division function
     * @param [in] aControl 1D view of control variables
     * @param [in] aTimeStep time step (default = 0.0)
     * @return scalar function evaluation
    **********************************************************************************/
    Plato::Scalar value(const Plato::ScalarVector & aControl) const
    {
        Plato::Scalar tNumeratorValue = mScalarFunctionBaseNumerator->value(aControl);
        Plato::Scalar tDenominatorValue = mScalarFunctionBaseDenominator->value(aControl);
        Plato::Scalar tResult = tNumeratorValue / tDenominatorValue;
        if (tDenominatorValue == 0.0)
        {
            THROWERR("Denominator of division function evaluated to 0!")
        }
        
        return tResult;
    }

    /******************************************************************************//**
     * @brief Evaluate gradient of the division function with respect to (wrt) the configuration parameters
     * @param [in] aControl 1D view of control variables
     * @param [in] aTimeStep time step (default = 0.0)
     * @return 1D view with the gradient of the scalar function wrt the configuration parameters
    **********************************************************************************/
    Plato::ScalarVector gradient_x(const Plato::ScalarVector & aControl) const
    {
        const Plato::OrdinalType tNumDofs = mNumSpatialDims * mNumNodes;
        Plato::ScalarVector tGradientX ("gradient configuration", tNumDofs);

        Plato::Scalar tNumeratorValue = mScalarFunctionBaseNumerator->value(aControl);
        Plato::Scalar tDenominatorValue = mScalarFunctionBaseDenominator->value(aControl);

        Plato::ScalarVector tNumeratorGradX = mScalarFunctionBaseNumerator->gradient_x(aControl);
        Plato::ScalarVector tDenominatorGradX = mScalarFunctionBaseDenominator->gradient_x(aControl);
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumDofs), LAMBDA_EXPRESSION(const Plato::OrdinalType & tDof)
        {
            tGradientX(tDof) = (tNumeratorGradX(tDof) * tDenominatorValue - 
                                tDenominatorGradX(tDof) * tNumeratorValue) 
                               / (pow(tDenominatorValue, 2));
        },"Division Function Grad X");
        return tGradientX;
    }

    /******************************************************************************//**
     * @brief Evaluate gradient of the division function with respect to (wrt) the control variables
     * @param [in] aControl 1D view of control variables
     * @param [in] aTimeStep time step (default = 0.0)
     * @return 1D view with the gradient of the scalar function wrt the control variables
    **********************************************************************************/
    Plato::ScalarVector gradient_z(const Plato::ScalarVector & aControl) const
    {
        const Plato::OrdinalType tNumDofs = mNumNodes;
        Plato::ScalarVector tGradientZ ("gradient control", tNumDofs);
        
        Plato::Scalar tNumeratorValue = mScalarFunctionBaseNumerator->value(aControl);
        Plato::Scalar tDenominatorValue = mScalarFunctionBaseDenominator->value(aControl);

        Plato::ScalarVector tNumeratorGradZ = mScalarFunctionBaseNumerator->gradient_z(aControl);
        Plato::ScalarVector tDenominatorGradZ = mScalarFunctionBaseDenominator->gradient_z(aControl);
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumDofs), LAMBDA_EXPRESSION(const Plato::OrdinalType & tDof)
        {
            tGradientZ(tDof) = (tNumeratorGradZ(tDof) * tDenominatorValue - 
                                tDenominatorGradZ(tDof) * tNumeratorValue) 
                               / (pow(tDenominatorValue, 2));
        },"Division Function Grad Z");

        return tGradientZ;
    }

    /******************************************************************************//**
     * @brief Set user defined function name
     * @param [in] function name
    **********************************************************************************/
    void setFunctionName(const std::string aFunctionName)
    {
        mFunctionName = aFunctionName;
    }

    /******************************************************************************//**
     * @brief Return user defined function name
     * @return User defined function name
    **********************************************************************************/
    std::string name() const
    {
        return mFunctionName;
    }
};
// class DivisionFunction

} // namespace Geometric

} // namespace Plato

#include "Geometrical.hpp"

#ifdef PLATOANALYZE_1D
extern template class Plato::Geometric::DivisionFunction<::Plato::Geometrical<1>>;
#endif

#ifdef PLATOANALYZE_2D
extern template class Plato::Geometric::DivisionFunction<::Plato::Geometrical<2>>;
#endif

#ifdef PLATOANALYZE_3D
extern template class Plato::Geometric::DivisionFunction<::Plato::Geometrical<3>>;
#endif
