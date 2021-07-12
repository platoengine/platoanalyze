#pragma once

#include <memory>
#include <cassert>
#include <vector>

#include <Omega_h_mesh.hpp>

#include "WorksetBase.hpp"
#include "PlatoStaticsTypes.hpp"
#include "elliptic/ScalarFunctionBase.hpp"
#include "elliptic/ScalarFunctionBaseFactory.hpp"
#include <Teuchos_ParameterList.hpp>

namespace Plato
{

namespace Elliptic
{

/******************************************************************************//**
 * \brief Division function class \f$ F(x) = numerator(x) / denominator(x) \f$
 **********************************************************************************/
template<typename PhysicsT>
class DivisionFunction : public Plato::Elliptic::ScalarFunctionBase, public Plato::WorksetBase<PhysicsT>
{
private:
    using Plato::WorksetBase<PhysicsT>::mNumDofsPerNode; /*!< number of degree of freedom per node */
    using Plato::WorksetBase<PhysicsT>::mNumSpatialDims; /*!< number of spatial dimensions */
    using Plato::WorksetBase<PhysicsT>::mNumNodes;       /*!< total number of nodes in the mesh */

    std::shared_ptr<Plato::Elliptic::ScalarFunctionBase> mScalarFunctionBaseNumerator; /*!< numerator function */
    std::shared_ptr<Plato::Elliptic::ScalarFunctionBase> mScalarFunctionBaseDenominator; /*!< denominator function */

    const Plato::SpatialModel & mSpatialModel;

    Plato::DataMap& mDataMap; /*!< PLATO Engine and Analyze data map */

    std::string mFunctionName; /*!< User defined function name */

	/******************************************************************************//**
     * \brief Initialization of Division Function
     * \param [in] aProblemParams input parameters database
    **********************************************************************************/
    void initialize(
        Teuchos::ParameterList & aProblemParams
    )
    {
        Plato::Elliptic::ScalarFunctionBaseFactory<PhysicsT> tFactory;

        auto tFunctionParams = aProblemParams.sublist("Criteria").sublist(mFunctionName);

        auto tNumeratorFunctionName = tFunctionParams.get<std::string>("Numerator");
        auto tDenominatorFunctionName = tFunctionParams.get<std::string>("Denominator");

        mScalarFunctionBaseNumerator = 
             tFactory.create(mSpatialModel, mDataMap, aProblemParams, tNumeratorFunctionName);

        mScalarFunctionBaseDenominator = 
             tFactory.create(mSpatialModel, mDataMap, aProblemParams, tDenominatorFunctionName);
    }

public:
    /******************************************************************************//**
     * \brief Primary division function constructor
     * \param [in] aSpatialModel Plato Analyze spatial model
     * \param [in] aDataMap Plato Analyze data map
     * \param [in] aInputParams input parameters database
     * \param [in] aName user defined function name
    **********************************************************************************/
    DivisionFunction(
        const Plato::SpatialModel    & aSpatialModel,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aProblemParams,
        const std::string            & aName
    ) :
        Plato::WorksetBase<PhysicsT>(aSpatialModel.Mesh),
        mSpatialModel (aSpatialModel),
        mDataMap      (aDataMap),
        mFunctionName (aName)
    {
        initialize(aProblemParams);
    }

    /******************************************************************************//**
     * \brief Secondary division function constructor, used for unit testing
     * \param [in] aMesh mesh database
     * \param [in] aMeshSets side sets database
    **********************************************************************************/
    DivisionFunction(
        const Plato::SpatialModel & aSpatialModel,
              Plato::DataMap      & aDataMap
    ) :
        Plato::WorksetBase<PhysicsT>(aSpatialModel.Mesh),
        mSpatialModel (aSpatialModel),
        mDataMap      (aDataMap),
        mFunctionName ("Division Function")
    {
    }

    /******************************************************************************//**
     * \brief Allocate numerator function base using the residual automatic differentiation type
     * \param [in] aInput scalar function
    **********************************************************************************/
    void allocateNumeratorFunction(const std::shared_ptr<Plato::Elliptic::ScalarFunctionBase>& aInput)
    {
        mScalarFunctionBaseNumerator = aInput;
    }

    /******************************************************************************//**
     * \brief Allocate denominator function base using the residual automatic differentiation type
     * \param [in] aInput scalar function
    **********************************************************************************/
    void allocateDenominatorFunction(const std::shared_ptr<Plato::Elliptic::ScalarFunctionBase>& aInput)
    {
        mScalarFunctionBaseDenominator = aInput;
    }

    /******************************************************************************//**
     * \brief Update physics-based parameters within optimization iterations
     * \param [in] aState 1D view of state variables
     * \param [in] aControl 1D view of control variables
     **********************************************************************************/
    void updateProblem(const Plato::ScalarVector & aState, const Plato::ScalarVector & aControl) const
    {
        mScalarFunctionBaseNumerator->updateProblem(aState, aControl);
        mScalarFunctionBaseDenominator->updateProblem(aState, aControl);
    }


    /******************************************************************************//**
     * \brief Evaluate division function
     * \param [in] aSolution solution database
     * \param [in] aControl 1D view of control variables
     * \param [in] aTimeStep time step (default = 0.0)
     * \return scalar function evaluation
    **********************************************************************************/
    Plato::Scalar
    value(const Plato::Solutions    & aSolution,
          const Plato::ScalarVector & aControl,
                Plato::Scalar         aTimeStep = 0.0) const override
    {
        Plato::Scalar tNumeratorValue = mScalarFunctionBaseNumerator->value(aSolution, aControl, aTimeStep);
        Plato::Scalar tDenominatorValue = mScalarFunctionBaseDenominator->value(aSolution, aControl, aTimeStep);
        Plato::Scalar tResult = tNumeratorValue / tDenominatorValue;
        if (tDenominatorValue == 0.0)
        {
            THROWERR("Denominator of division function evaluated to 0!")
        }
        
        return tResult;
    }

    /******************************************************************************//**
     * \brief Evaluate gradient of the division function with respect to (wrt) the configuration parameters
     * \param [in] aSolution solution database
     * \param [in] aControl 1D view of control variables
     * \param [in] aTimeStep time step (default = 0.0)
     * \return 1D view with the gradient of the scalar function wrt the configuration parameters
    **********************************************************************************/
    Plato::ScalarVector
    gradient_x(const Plato::Solutions    & aSolution,
               const Plato::ScalarVector & aControl,
                     Plato::Scalar         aTimeStep = 0.0) const override
    {
        const Plato::OrdinalType tNumDofs = mNumSpatialDims * mNumNodes;
        Plato::ScalarVector tGradientX ("gradient configuration", tNumDofs);

        Plato::Scalar tNumeratorValue = mScalarFunctionBaseNumerator->value(aSolution, aControl, aTimeStep);
        Plato::Scalar tDenominatorValue = mScalarFunctionBaseDenominator->value(aSolution, aControl, aTimeStep);
        Plato::Scalar tDenominatorValueSquared = tDenominatorValue * tDenominatorValue;

        Plato::ScalarVector tNumeratorGradX = mScalarFunctionBaseNumerator->gradient_x(aSolution, aControl, aTimeStep);
        Plato::ScalarVector tDenominatorGradX = mScalarFunctionBaseDenominator->gradient_x(aSolution, aControl, aTimeStep);
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumDofs), LAMBDA_EXPRESSION(const Plato::OrdinalType & tDof)
        {
            tGradientX(tDof) = (tNumeratorGradX(tDof) * tDenominatorValue - 
                                tDenominatorGradX(tDof) * tNumeratorValue) 
                               / (tDenominatorValueSquared);
        },"Division Function Grad X");
        return tGradientX;
    }

    /******************************************************************************//**
     * \brief Evaluate gradient of the division function with respect to (wrt) the state variables
     * \param [in] aSolution solution database
     * \param [in] aControl 1D view of control variables
     * \param [in] aTimeStep time step (default = 0.0)
     * \return 1D view with the gradient of the scalar function wrt the state variables
    **********************************************************************************/
    Plato::ScalarVector
    gradient_u(const Plato::Solutions    & aSolution,
               const Plato::ScalarVector & aControl,
                     Plato::OrdinalType    aStepIndex,
                     Plato::Scalar         aTimeStep = 0.0) const override
    {
        const Plato::OrdinalType tNumDofs = mNumDofsPerNode * mNumNodes;
        Plato::ScalarVector tGradientU ("gradient state", tNumDofs);

        Plato::Scalar tNumeratorValue = mScalarFunctionBaseNumerator->value(aSolution, aControl, aTimeStep);
        Plato::Scalar tDenominatorValue = mScalarFunctionBaseDenominator->value(aSolution, aControl, aTimeStep);
        Plato::Scalar tDenominatorValueSquared = tDenominatorValue * tDenominatorValue;

        Plato::ScalarVector tNumeratorGradU = mScalarFunctionBaseNumerator->gradient_u(aSolution, aControl, aTimeStep);
        Plato::ScalarVector tDenominatorGradU = mScalarFunctionBaseDenominator->gradient_u(aSolution, aControl, aTimeStep);
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumDofs), LAMBDA_EXPRESSION(const Plato::OrdinalType & tDof)
        {
            tGradientU(tDof) = (tNumeratorGradU(tDof) * tDenominatorValue - 
                                tDenominatorGradU(tDof) * tNumeratorValue) 
                               / (tDenominatorValueSquared);
        },"Division Function Grad U");

        return tGradientU;
    }

    /******************************************************************************//**
     * \brief Evaluate gradient of the division function with respect to (wrt) the control variables
     * \param [in] aSolution solution database
     * \param [in] aControl 1D view of control variables
     * \param [in] aTimeStep time step (default = 0.0)
     * \return 1D view with the gradient of the scalar function wrt the control variables
    **********************************************************************************/
    Plato::ScalarVector
    gradient_z(const Plato::Solutions    & aSolution,
               const Plato::ScalarVector & aControl,
                     Plato::Scalar aTimeStep = 0.0) const override
    {
        const Plato::OrdinalType tNumDofs = mNumNodes;
        Plato::ScalarVector tGradientZ ("gradient control", tNumDofs);
        
        Plato::Scalar tNumeratorValue = mScalarFunctionBaseNumerator->value(aSolution, aControl, aTimeStep);
        Plato::Scalar tDenominatorValue = mScalarFunctionBaseDenominator->value(aSolution, aControl, aTimeStep);
        Plato::Scalar tDenominatorValueSquared = tDenominatorValue * tDenominatorValue;

        Plato::ScalarVector tNumeratorGradZ = mScalarFunctionBaseNumerator->gradient_z(aSolution, aControl, aTimeStep);
        Plato::ScalarVector tDenominatorGradZ = mScalarFunctionBaseDenominator->gradient_z(aSolution, aControl, aTimeStep);
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumDofs), LAMBDA_EXPRESSION(const Plato::OrdinalType & tDof)
        {
            tGradientZ(tDof) = (tNumeratorGradZ(tDof) * tDenominatorValue - 
                                tDenominatorGradZ(tDof) * tNumeratorValue) 
                               / (tDenominatorValueSquared);
        },"Division Function Grad Z");

        return tGradientZ;
    }

    /******************************************************************************//**
     * \brief Set user defined function name
     * \param [in] function name
    **********************************************************************************/
    void setFunctionName(const std::string aFunctionName)
    {
        mFunctionName = aFunctionName;
    }

    /******************************************************************************//**
     * \brief Return user defined function name
     * \return User defined function name
    **********************************************************************************/
    std::string name() const
    {
        return mFunctionName;
    }
};
// class DivisionFunction

} // namespace Elliptic

} // namespace Plato

#include "Thermal.hpp"
#include "Mechanics.hpp"
#include "Electromechanics.hpp"
#include "Thermomechanics.hpp"

#ifdef PLATOANALYZE_1D
extern template class Plato::Elliptic::DivisionFunction<::Plato::Thermal<1>>;
extern template class Plato::Elliptic::DivisionFunction<::Plato::Mechanics<1>>;
extern template class Plato::Elliptic::DivisionFunction<::Plato::Electromechanics<1>>;
extern template class Plato::Elliptic::DivisionFunction<::Plato::Thermomechanics<1>>;
#endif

#ifdef PLATOANALYZE_2D
extern template class Plato::Elliptic::DivisionFunction<::Plato::Thermal<2>>;
extern template class Plato::Elliptic::DivisionFunction<::Plato::Mechanics<2>>;
extern template class Plato::Elliptic::DivisionFunction<::Plato::Electromechanics<2>>;
extern template class Plato::Elliptic::DivisionFunction<::Plato::Thermomechanics<2>>;
#endif

#ifdef PLATOANALYZE_3D
extern template class Plato::Elliptic::DivisionFunction<::Plato::Thermal<3>>;
extern template class Plato::Elliptic::DivisionFunction<::Plato::Mechanics<3>>;
extern template class Plato::Elliptic::DivisionFunction<::Plato::Electromechanics<3>>;
extern template class Plato::Elliptic::DivisionFunction<::Plato::Thermomechanics<3>>;
#endif
