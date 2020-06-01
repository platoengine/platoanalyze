#pragma once

#include "PlatoStaticsTypes.hpp"

namespace Plato
{

namespace Parabolic
{

/******************************************************************************/
template<typename EvaluationType>
class TrapezoidIntegrator
/******************************************************************************/
{
    Plato::Scalar mAlpha;

public:
    /******************************************************************************/
    explicit 
    TrapezoidIntegrator(Teuchos::ParameterList& aParams) :
      mAlpha( aParams.get<double>("Trapezoid Alpha") )
    /******************************************************************************/
    {
    }
    /******************************************************************************/
    ~TrapezoidIntegrator()
    /******************************************************************************/
    {
    }

    /******************************************************************************/
    Plato::Scalar
    v_grad_u( Plato::Scalar aTimeStep )
    /******************************************************************************/
    {
        return -1.0/(mAlpha*aTimeStep);
    }

    /******************************************************************************/
    Plato::Scalar
    v_grad_u_prev( Plato::Scalar aTimeStep )
    /******************************************************************************/
    {
        return 1.0/(mAlpha*aTimeStep);
    }

    /******************************************************************************/
    Plato::Scalar
    v_grad_v_prev( Plato::Scalar aTimeStep )
    /******************************************************************************/
    {
        return (1.0-mAlpha)/mAlpha;
    }


    /******************************************************************************/
    Plato::ScalarVector 
    v_value(const Plato::ScalarVector & aU,
            const Plato::ScalarVector & aU_prev,
            const Plato::ScalarVector & aV,
            const Plato::ScalarVector & aV_prev,
                  Plato::Scalar dt)
    /******************************************************************************/
    {
        auto tNumData = aU.extent(0);
        Plato::ScalarVector tReturnValue("velocity residual", tNumData);

        auto tAlpha = mAlpha;
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumData), LAMBDA_EXPRESSION(const Plato::OrdinalType & aOrdinal)
        {
            tReturnValue(aOrdinal) = aV(aOrdinal) - 1.0/(tAlpha*dt)*(aU(aOrdinal) - aU_prev(aOrdinal) - dt*(1.0-tAlpha)*aV_prev(aOrdinal));
        }, "Velocity residual value");

        return tReturnValue;
    }
};

} // namespace Parabolic

} // namespace Plato
