#ifndef NEWMARK_HPP
#define NEWMARK_HPP

#include "PlatoStaticsTypes.hpp"

namespace Plato
{

/******************************************************************************/
template<typename EvaluationType>
class NewmarkIntegrator
/******************************************************************************/
{
    Plato::Scalar mGamma;
    Plato::Scalar mBeta;
public:
    /******************************************************************************/
    explicit 
    NewmarkIntegrator(Teuchos::ParameterList& aParams) :
      mGamma( aParams.get<double>("Newmark Gamma") ),
      mBeta ( aParams.get<double>("Newmark Beta") )
    /******************************************************************************/
    {
    }
    /******************************************************************************/
    ~NewmarkIntegrator()
    /******************************************************************************/
    {
    }

    /******************************************************************************/
    Plato::Scalar
    v_grad_u( Plato::Scalar aTimeStep )
    /******************************************************************************/
    {
        return -mGamma/(mBeta*aTimeStep);
    }

    /******************************************************************************/
    Plato::Scalar
    v_grad_u_prev( Plato::Scalar aTimeStep )
    /******************************************************************************/
    {
        return mGamma/(mBeta*aTimeStep);
    }

    /******************************************************************************/
    Plato::Scalar
    v_grad_v_prev( Plato::Scalar aTimeStep )
    /******************************************************************************/
    {
        return mGamma/mBeta - 1.0;
    }

    /******************************************************************************/
    Plato::Scalar
    v_grad_a_prev( Plato::Scalar aTimeStep )
    /******************************************************************************/
    {
        return (mGamma/(2.0*mBeta) - 1.0) * aTimeStep;
    }

    /******************************************************************************/
    Plato::Scalar
    a_grad_u( Plato::Scalar aTimeStep )
    /******************************************************************************/
    {
        return -1.0/(mBeta*aTimeStep*aTimeStep);
    }

    /******************************************************************************/
    Plato::Scalar
    a_grad_u_prev( Plato::Scalar aTimeStep )
    /******************************************************************************/
    {
        return 1.0/(mBeta*aTimeStep*aTimeStep);
    }

    /******************************************************************************/
    Plato::Scalar
    a_grad_v_prev( Plato::Scalar aTimeStep )
    /******************************************************************************/
    {
        return 1.0/(mBeta*aTimeStep);
    }

    /******************************************************************************/
    Plato::Scalar
    a_grad_a_prev( Plato::Scalar aTimeStep )
    /******************************************************************************/
    {
        return 1.0/(2.0*mBeta) - 1.0;
    }

    /******************************************************************************/
    Plato::ScalarVector 
    v_value(const Plato::ScalarVector & aU,
            const Plato::ScalarVector & aU_prev,
            const Plato::ScalarVector & aV,
            const Plato::ScalarVector & aV_prev,
            const Plato::ScalarVector & aA,
            const Plato::ScalarVector & aA_prev,
                  Plato::Scalar dt)
    /******************************************************************************/
    {
        auto tNumData = aU.extent(0);
        Plato::ScalarVector tReturnValue("velocity residual", tNumData);

        auto tGamma = mGamma;
        auto tBeta = mBeta;
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumData), LAMBDA_EXPRESSION(const Plato::OrdinalType & aOrdinal)
        {
            Plato::Scalar tPredV = aV_prev(aOrdinal) + (1.0-tGamma)*dt*aA_prev(aOrdinal);
            Plato::Scalar tPredU = aU_prev(aOrdinal) + dt*aV_prev(aOrdinal) + dt*dt/2.0*(1.0-2.0*tBeta)* aA_prev(aOrdinal);
            tReturnValue(aOrdinal) = aV(aOrdinal) - tPredV - tGamma/(tBeta*dt)*(aU(aOrdinal) - tPredU);
        }, "Velocity residual value");

        return tReturnValue;
    }

    /******************************************************************************/
    Plato::ScalarVector 
    a_value(const Plato::ScalarVector & aU,
            const Plato::ScalarVector & aU_prev,
            const Plato::ScalarVector & aV,
            const Plato::ScalarVector & aV_prev,
            const Plato::ScalarVector & aA,
            const Plato::ScalarVector & aA_prev,
                  Plato::Scalar dt)
    /******************************************************************************/
    {
        auto tNumData = aU.extent(0);
        Plato::ScalarVector tReturnValue("velocity residual", tNumData);

        auto tBeta = mBeta;
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumData), LAMBDA_EXPRESSION(const Plato::OrdinalType & aOrdinal)
        {
            Plato::Scalar tPredU = aU_prev(aOrdinal) + dt*aV_prev(aOrdinal) + dt*dt/2.0*(1.0-2.0*tBeta)* aA_prev(aOrdinal);
            tReturnValue(aOrdinal) = aA(aOrdinal) - 1.0/(tBeta*dt*dt)*(aU(aOrdinal) - tPredU);
        }, "Velocity residual value");

        return tReturnValue;
    }
};

} // namespace Plato

#endif
