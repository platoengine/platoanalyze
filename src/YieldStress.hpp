#ifndef PLATO_YEILD_STRESS_HPP
#define PLATO_YEILD_STRESS_HPP

#include "AbstractYieldStress.hpp"

namespace Plato
{

/******************************************************************************/
/*! Yield Stress functor.
 *
 * \tparam EvaluationType - the evaluation type
 */
/******************************************************************************/
template<typename EvaluationType>
class YieldStress :
    public Plato::AbstractYieldStress<EvaluationType >
{
protected:
    using LocalStateT = typename EvaluationType::LocalStateScalarType; /*!< local state variables automatic differentiation type */
    using ControlT    = typename EvaluationType::ControlScalarType;    /*!< control variables automatic differentiation type */
    using ResultT     = typename EvaluationType::ResultScalarType;     /*!< result variables automatic differentiation type */

public:
    /******************************************************************************//**
     * \brief Constructor
    **********************************************************************************/
    YieldStress() {}

    /******************************************************************************//**
     * \brief Compute the yield stress
     * \param [in]  aCellOrdinal element ordinal
     * \param [out] aYieldStress - yield stress
     * \param [in]  aLocalState
     * \param [in]  aPenalizedInitialYieldStress,
     * \param [in]  aPenalizedHardeningModulusIsotropic
    **********************************************************************************/
    DEVICE_TYPE inline void
    operator()(Plato::OrdinalType const & aCellOrdinal,
               ResultT & aYieldStress,
               Plato::ScalarMultiVectorT< LocalStateT > const& aLocalState,
               ControlT const& aPenalizedInitialYieldStress,
               ControlT const& aPenalizedHardeningModulusIsotropic) const
    {
        // Method used to compute the stress and called from within a
        // Kokkos parallel_for when there is a single return value

        // Currently not used (was initally used when devloping the
        // yield stress paradigm).

        // Compute the yield stress for a single value.
        aYieldStress =
          aPenalizedInitialYieldStress +
          aPenalizedHardeningModulusIsotropic *
          aLocalState(aCellOrdinal, 0); // SHOULD THIS BE PREV? I think no.
    }

    /******************************************************************************//**
     * \brief Compute the yield stress
     * \param [in]  aCellOrdinal element ordinal
     * \param [out] aYieldStress - yield stress
     * \param [in]  aLocalState
     * \param [in]  aPenalizedInitialYieldStress,
     * \param [in]  aPenalizedHardeningModulusIsotropic
    **********************************************************************************/
    DEVICE_TYPE inline void
    operator()(Plato::OrdinalType const & aCellOrdinal,
               Plato::ScalarMultiVectorT< ResultT     > const& aYieldStress,
               Plato::ScalarMultiVectorT< LocalStateT > const& aLocalState,
               Plato::ScalarVectorT< ControlT > const& aPenalizedInitialYieldStress,
               Plato::ScalarVectorT< ControlT > const& aPenalizedHardeningModulusIsotropic) const
    {
        // Method used to compute the stress and called from within a
        // Kokkos parallel_for when there are multiple return values.

        // Currently not used (was initally used when devloping the
        // yield stress paradigm).

        // Compute the yield stress for a single value in a view.
        aYieldStress(aCellOrdinal, 0) =
          aPenalizedInitialYieldStress(aCellOrdinal) +
          aPenalizedHardeningModulusIsotropic(aCellOrdinal) *
          aLocalState(aCellOrdinal, 0); // SHOULD THIS BE PREV? I think no.
    }

    /******************************************************************************//**
     * \brief Compute the yield stress
     * \param [out] tYieldStress - yield stress
     * \param [in]  aLocalState
     * \param [in]  aParameters
    **********************************************************************************/
    void
    operator()(Plato::ScalarMultiVectorT< ResultT     > const& aYieldStress,
               Plato::ScalarMultiVectorT< LocalStateT > const& aLocalState,
               Kokkos::View< Plato::ScalarVectorT< ControlT > *,
                             Kokkos::CudaUVMSpace > const& aParameters) const override
    {
        // Method used to compute the stress with the factory and has
        // its own Kokkos parallel_for.
        Plato::OrdinalType tNumCells = aYieldStress.extent(0);

        // There must three parameters, including the Local State.
        if( aParameters.extent(0) != 2 )
        {
          std::stringstream errorMsg;
          errorMsg << "Calling the original yield stress equation with an "
                 << "incorrect number of parameters, expected 2, received "
                   << aParameters.extent(0) << ".";

          THROWERR( errorMsg.str() );
        }

        // For ease get the parameters based on their true names.
        const auto & aPenalizedInitialYieldStress        = aParameters(0);
        const auto & aPenalizedHardeningModulusIsotropic = aParameters(1);

        // Compute the yield stress.
        Kokkos::parallel_for( "Compute yield stress",
                              Kokkos::RangePolicy<>(0, tNumCells),
                              LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
        {
          aYieldStress(aCellOrdinal, 0) =
            aPenalizedInitialYieldStress(aCellOrdinal) +
            aPenalizedHardeningModulusIsotropic(aCellOrdinal) *
            aLocalState(aCellOrdinal, 0); // SHOULD THIS BE PREV? I think no.
        } );

        // Drop all of the refernces to the parameter data.
        for( Plato::OrdinalType i=0; i<aParameters.extent(0)-1; ++i )
        {
          aParameters(i) = Plato::ScalarVectorT< ControlT >( "Yield Stress Dummy Parameter" );
        }
    }
};
// class YieldStress

}// namespace Plato
#endif

#ifdef PLATOANALYZE_2D
PLATO_EXPL_DEC_INC_LOCAL_1(Plato::YieldStress,           Plato::SimplexPlasticity,       2)
PLATO_EXPL_DEC_INC_LOCAL_1(Plato::YieldStress,           Plato::SimplexThermoPlasticity, 2)
#endif

#ifdef PLATOANALYZE_3D
PLATO_EXPL_DEC_INC_LOCAL_1(Plato::YieldStress,           Plato::SimplexPlasticity,       3)
PLATO_EXPL_DEC_INC_LOCAL_1(Plato::YieldStress,           Plato::SimplexThermoPlasticity, 3)
#endif
