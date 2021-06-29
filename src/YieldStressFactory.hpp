#ifndef PLATO_YEILD_STRESS_FACTORY_HPP
#define PLATO_YEILD_STRESS_FACTORY_HPP

#include "YieldStress.hpp"
#include "YieldStressExpression.hpp"

namespace Plato
{
/******************************************************************************//**
v * \brief Yield Stress Factory for creating yield stress models.
 *
 * \tparam EvaluationType - the evaluation type
 *
**********************************************************************************/
template<typename EvaluationType>
class YieldStressFactory
{
public:
    /******************************************************************************//**
    * \brief yield stress factory constructor.
    **********************************************************************************/
    YieldStressFactory() {}

    /******************************************************************************//**
    * \brief Create a yield stress functor.
    * \param [in] const aParamList input parameter list
    * \return Teuchos reference counter pointer to yield stress functor
    **********************************************************************************/
    Teuchos::RCP<Plato::AbstractYieldStress<EvaluationType> > create(
        const Teuchos::ParameterList& mParamList)
    {
      // Look for a yield stress block.
      if( mParamList.isSublist("Custom Plasticity Model") )
      {
        return Teuchos::rcp( new Plato::YieldStressExpression<EvaluationType>
                             (mParamList) );
      }
      else
      {
        return Teuchos::rcp( new Plato::YieldStress<EvaluationType> );
      }
    }
};
// class YieldStressFactory

}// namespace Plato
#endif

#ifdef PLATOANALYZE_2D
PLATO_EXPL_DEC_INC_LOCAL_1(Plato::YieldStressFactory,    Plato::SimplexPlasticity,       2)
PLATO_EXPL_DEC_INC_LOCAL_1(Plato::YieldStressFactory,    Plato::SimplexThermoPlasticity, 2)
#endif

#ifdef PLATOANALYZE_3D
PLATO_EXPL_DEC_INC_LOCAL_1(Plato::YieldStressFactory,    Plato::SimplexPlasticity,       3)
PLATO_EXPL_DEC_INC_LOCAL_1(Plato::YieldStressFactory,    Plato::SimplexThermoPlasticity, 3)
#endif
