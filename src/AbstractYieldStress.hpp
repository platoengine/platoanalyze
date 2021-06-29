#ifndef PLATO_ABSTRACT_YEILD_STRESS_HPP
#define PLATO_ABSTRACT_YEILD_STRESS_HPP

#include "ExpInstMacros.hpp"

#include "SimplexFadTypes.hpp"
#include "SimplexPlasticity.hpp"
#include "SimplexThermoPlasticity.hpp"

namespace Plato
{
/******************************************************************************/
/*! Abstract Yield Stress functor.
 *
 * \tparam EvaluationType - the evaluation type
 */
/******************************************************************************/
template<typename EvaluationType>
class AbstractYieldStress
{
protected:
    using LocalStateT = typename EvaluationType::LocalStateScalarType; /*!< local state variables automatic differentiation type */
    using ControlT    = typename EvaluationType::ControlScalarType;    /*!< control variables automatic differentiation type */
    using ResultT     = typename EvaluationType::ResultScalarType;     /*!< result variables automatic differentiation type */

public:
    /******************************************************************************//**
     * \brief Constructor
    **********************************************************************************/
    AbstractYieldStress() {}

    /******************************************************************************//**
     * \brief Compute the yield stress
     * \param [out] aResult - yield stress
     * \param [in]  aLocalState
     * \param [in]  aArgs ... one or more
    **********************************************************************************/
    void
    operator()(Plato::ScalarMultiVectorT< ResultT     > const& aResult,
               Plato::ScalarMultiVectorT< LocalStateT > const& aLocalState) const
    {
      // Plato::OrdinalType tNumParamNames = 0;

      // Create a view to point to each of the parameters being passed
      // in. Note: all parameters are assumed to be of type
      // 'Plato::ScalarVectorT< ControlT >'.
      Kokkos::View< Plato::ScalarVectorT< ControlT > *, Kokkos::CudaUVMSpace >
        tParameters("Yield Stress Parameters");

      operator()(aResult, aLocalState, tParameters);
    }

    // The LocalState plus one parameter.
    void
    operator()(Plato::ScalarMultiVectorT< ResultT     > const& aResult,
               Plato::ScalarMultiVectorT< LocalStateT > const& aLocalState,
               Plato::ScalarVectorT< ControlT > const& aArg0) const
    {
      Plato::OrdinalType tNumParamNames = 1;

      // Create a view to point to each of the parameters being passed
      // in. Note: all parameters are assumed to be of type
      // 'Plato::ScalarVectorT< ControlT >'.
      Kokkos::View< Plato::ScalarVectorT< ControlT > *, Kokkos::CudaUVMSpace >
        tParameters("Yield Stress Parameters", tNumParamNames);

      tParameters(0) = aArg0;

      operator()(aResult, aLocalState, tParameters);
    }

    // The LocalState plus two parameters.
    void
    operator()(Plato::ScalarMultiVectorT< ResultT     > const& aResult,
               Plato::ScalarMultiVectorT< LocalStateT > const& aLocalState,
               Plato::ScalarVectorT< ControlT > const& aArg0,
               Plato::ScalarVectorT< ControlT > const& aArg1) const
    {
      Plato::OrdinalType tNumParamNames = 2;

      // Create a view to point to each of the parameters being passed
      // in. Note: all parameters are assumed to be of type
      // 'Plato::ScalarVectorT< ControlT >'.
      Kokkos::View< Plato::ScalarVectorT< ControlT > *, Kokkos::CudaUVMSpace >
        tParameters("Yield Stress Parameters", tNumParamNames);

      tParameters(0) = aArg0;
      tParameters(1) = aArg1;

      operator()(aResult, aLocalState, tParameters);
    }

    // The LocalState plus three parameters.
    void
    operator()(Plato::ScalarMultiVectorT< ResultT     > const& aResult,
               Plato::ScalarMultiVectorT< LocalStateT > const& aLocalState,
               Plato::ScalarVectorT< ControlT > const& aArg0,
               Plato::ScalarVectorT< ControlT > const& aArg1,
               Plato::ScalarVectorT< ControlT > const& aArg2) const
    {
      Plato::OrdinalType tNumParamNames = 3;

      // Create a view to point to each of the parameters being passed
      // in. Note: all parameters are assumed to be of type
      // 'Plato::ScalarVectorT< ControlT >'.
      Kokkos::View< Plato::ScalarVectorT< ControlT > *, Kokkos::CudaUVMSpace >
        tParameters("Yield Stress Parameters", tNumParamNames);

      tParameters(0) = aArg0;
      tParameters(1) = aArg1;
      tParameters(2) = aArg2;

      operator()(aResult, aLocalState, tParameters);
    }

    // The LocalState plus four parameters.
    void
    operator()(Plato::ScalarMultiVectorT< ResultT     > const& aResult,
               Plato::ScalarMultiVectorT< LocalStateT > const& aLocalState,
               Plato::ScalarVectorT< ControlT > const& aArg0,
               Plato::ScalarVectorT< ControlT > const& aArg1,
               Plato::ScalarVectorT< ControlT > const& aArg2,
               Plato::ScalarVectorT< ControlT > const& aArg3) const
    {
      Plato::OrdinalType tNumParamNames = 4;

      // Create a view to point to each of the parameters being passed
      // in. Note: all parameters are assumed to be of type
      // 'Plato::ScalarVectorT< ControlT >'.
      Kokkos::View< Plato::ScalarVectorT< ControlT > *, Kokkos::CudaUVMSpace >
        tParameters("Yield Stress Parameters", tNumParamNames);

      // The first is reserved for the LocalState.
      tParameters(0) = aArg0;
      tParameters(1) = aArg1;
      tParameters(2) = aArg2;
      tParameters(3) = aArg3;

      operator()(aResult, aLocalState, tParameters);
    }

    /******************************************************************************//**
     * \brief Compute the yield stress
     * \param [out] aResult - yield stress
     * \param [in]  aLocalState
     * \param [in]  aParameters
    **********************************************************************************/
    virtual void
    operator()(Plato::ScalarMultiVectorT< ResultT     > const& aResult,
               Plato::ScalarMultiVectorT< LocalStateT > const& aLocalState,
               Kokkos::View< Plato::ScalarVectorT< ControlT > *,
                             Kokkos::CudaUVMSpace > const& aParameters) const = 0;

};
// class AbstractYieldStress

}// namespace Plato
#endif

#ifdef PLATOANALYZE_2D
PLATO_EXPL_DEC_INC_LOCAL_1(Plato::AbstractYieldStress,   Plato::SimplexPlasticity,       2)
PLATO_EXPL_DEC_INC_LOCAL_1(Plato::AbstractYieldStress,   Plato::SimplexThermoPlasticity, 2)
#endif

#ifdef PLATOANALYZE_3D
PLATO_EXPL_DEC_INC_LOCAL_1(Plato::AbstractYieldStress,   Plato::SimplexPlasticity,       3)
PLATO_EXPL_DEC_INC_LOCAL_1(Plato::AbstractYieldStress,   Plato::SimplexThermoPlasticity, 3)
#endif
