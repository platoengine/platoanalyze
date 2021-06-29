#ifndef PLATO_LINEAR_STRESS_FACTORY_HPP
#define PLATO_LINEAR_STRESS_FACTORY_HPP

#include <LinearStress.hpp>
#include <LinearStressExpression.hpp>

namespace Plato
{
/******************************************************************************//**
 * \brief Linear Stress Factory for creating linear stress models.
 *
 * \tparam EvaluationType - the evaluation type
 *
**********************************************************************************/
template<typename EvaluationType, typename SimplexPhysics>
class LinearStressFactory
{
public:
    /******************************************************************************//**
    * \brief linear stress factory constructor.
    **********************************************************************************/
    LinearStressFactory() {}

    /******************************************************************************//**
    * \brief Create a linear stress functor.
    * \param [in] aMaterialInfo - a material element stiffness matrix or
                                  a material model interface
    * \param [in] aParamList - input parameter list
    * \return Teuchos reference counter pointer to the linear stress functor
    **********************************************************************************/
    template<typename MaterialInfoType >
    Teuchos::RCP<Plato::AbstractLinearStress<EvaluationType,
                                             SimplexPhysics> > create(
        const MaterialInfoType aMaterialInfo,
        const Teuchos::ParameterList& aParamList)
    {
      // Look for a linear stress block.
      if( aParamList.isSublist("Custom Elasticity Model") )
      {
        return Teuchos::rcp( new Plato::LinearStressExpression<EvaluationType,
                                                               SimplexPhysics>
                             (aMaterialInfo, aParamList) );
      }
      else
      {
        return Teuchos::rcp( new Plato::LinearStress<EvaluationType,
                                                     SimplexPhysics>
                             (aMaterialInfo) );
      }
    }
};
// class LinearStressFactory

}// namespace Plato
#endif

#ifdef PLATOANALYZE_1D
PLATO_EXPL_DEC2(Plato::LinearStressFactory, Plato::SimplexMechanics, 1)
#endif

#ifdef PLATOANALYZE_2D
PLATO_EXPL_DEC2(Plato::LinearStressFactory, Plato::SimplexMechanics, 2)
#endif

#ifdef PLATOANALYZE_3D
PLATO_EXPL_DEC2(Plato::LinearStressFactory, Plato::SimplexMechanics, 3)
#endif
