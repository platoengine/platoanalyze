#ifndef PLATO_HYPERBOLIC_LINEAR_STRESS_FACTORY_HPP
#define PLATO_HYPERBOLIC_LINEAR_STRESS_FACTORY_HPP

#include <hyperbolic/HyperbolicExpInstMacros.hpp>
#include <hyperbolic/HyperbolicLinearStress.hpp>
#include <hyperbolic/HyperbolicLinearStressExpression.hpp>

namespace Plato
{

namespace Hyperbolic
{

/******************************************************************************//**
 * \brief Linear Stress Factory for creating linear stress models.
 *
 * \tparam EvaluationType - the evaluation type
 *
**********************************************************************************/
template<typename EvaluationType, typename SimplexPhysics>
class HyperbolicLinearStressFactory
{
public:
    /******************************************************************************//**
    * \brief linear stress factory constructor.
    **********************************************************************************/
    HyperbolicLinearStressFactory() {}

    /******************************************************************************//**
    * \brief Create a hyperbolic linear stress functor.
    * \param [in] aMaterialInfo - a material element stiffness matrix or
                                  a material model interface
    * \param [in] aParamList - input parameter list
    * \return Teuchos reference counter pointer to the linear stress functor
    **********************************************************************************/
    template<typename MaterialInfoType >
    Teuchos::RCP<Plato::Hyperbolic::HyperbolicAbstractLinearStress<EvaluationType,
								   SimplexPhysics> > create(
        const MaterialInfoType aMaterialInfo,
        const Teuchos::ParameterList& aParamList)
    {
      // Look for a linear stress block.
      if( aParamList.isSublist("Custom Elasticity Model") )
      {
          return Teuchos::rcp( new
              Plato::Hyperbolic::HyperbolicLinearStressExpression<EvaluationType,
                                                                  SimplexPhysics>
                               (aMaterialInfo, aParamList) );
      }
      else
      {
          return Teuchos::rcp( new
              Plato::Hyperbolic::HyperbolicLinearStress<EvaluationType,
                                                        SimplexPhysics>
                               (aMaterialInfo) );
      }
    }
};
// class HyperbolicLinearStressFactory

}// namespace Hyperbolic

}// namespace Plato
#endif

#ifdef PLATOANALYZE_1D
PLATO_HYPERBOLIC_EXPL_DEC2(Plato::Hyperbolic::HyperbolicLinearStressFactory, Plato::SimplexMechanics, 1)
#endif

#ifdef PLATOANALYZE_2D
PLATO_HYPERBOLIC_EXPL_DEC2(Plato::Hyperbolic::HyperbolicLinearStressFactory, Plato::SimplexMechanics, 2)
#endif

#ifdef PLATOANALYZE_3D
PLATO_HYPERBOLIC_EXPL_DEC2(Plato::Hyperbolic::HyperbolicLinearStressFactory, Plato::SimplexMechanics, 3)
#endif
