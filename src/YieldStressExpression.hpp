#ifndef PLATO_YEILD_STRESS_EXPRESSION_HPP
#define PLATO_YEILD_STRESS_EXPRESSION_HPP

#include "AbstractYieldStress.hpp"

#include "ExpressionEvaluator.hpp"
#include "ParseTools.hpp"

/******************************************************************************/
/*
 To use the expression evaluator one must first add an expression to
 the analyzeInput.xml file as part of the 'Custom Plasticity Model':

      <ParameterList name='Custom Plasticity Model'>
        <Parameter  name='Equation' type='string' value='PenalizedHardeningModulusIsotropic * AccumulatedPlasticStrain + PenalizedInitialYieldStress'/>

        <Parameter  name='AccumulatedPlasticStrain' type='string' value='Local State Workset'/>

 Here the equation variable names, PenalizedHardeningModulusIsotropic
 and PenalizedInitialYieldStress are directly mapped to the parameter
 labels which are set as part of the Kokkos::view.

 Whereas the equation variable name, AccumulatedPlasticStrain is
 indirectly mapped to the parameter label for the 'Local State
 Workset' because the parameter label contains whitespace and
 variables names cannot.

 aLocalState must be passed in to the operator() regardless it if used
 or not. Whereas parameters are optional, currently zero to four may
 be passed in to the operator().

 Equation variables can also be fixed values:
        <Parameter  name='PHMI' type='double' value='0.01'/>

 The equation can also be from a Bingo file:

      <ParameterList name='Custom Plasticity Model'>
        <Parameter name="BingoFile" type="string" value="bingo.txt"/>

*/
/******************************************************************************/

namespace Plato
{
/******************************************************************************/
/*! Yield Stress Expression functor.
 *
 * \tparam EvaluationType - the evaluation type
v */
/******************************************************************************/
template<typename EvaluationType>
class YieldStressExpression :
    public Plato::AbstractYieldStress<EvaluationType>
{
protected:
    using LocalStateT = typename EvaluationType::LocalStateScalarType; /*!< local state variables automatic differentiation type */
    using ControlT    = typename EvaluationType::ControlScalarType;    /*!< control variables automatic differentiation type */
    using ResultT     = typename EvaluationType::ResultScalarType;     /*!< result variables automatic differentiation type */

    Teuchos::ParameterList mInputParams;

public:
// ************************************************************************* //
    // Map structure - used with Kokkos so char strings so to be compatable.
    template< typename KEY_TYPE, typename VALUE_TYPE > struct _Map {
      KEY_TYPE key;
      VALUE_TYPE value;
    };

    template< typename KEY_TYPE, typename VALUE_TYPE >
    using Map = _Map< KEY_TYPE, VALUE_TYPE>;

    using VariableMap = Map< Plato::OrdinalType, char[MAX_ARRAY_LENGTH] >;

    /******************************************************************************//**
     * \brief Constructor
     * \param [in] aInputParams Teuchos parameter list
    **********************************************************************************/
    YieldStressExpression(const Teuchos::ParameterList& aInputParams) :
      mInputParams(aInputParams)
    {
    }

    /******************************************************************************//**
     * \brief Compute the yield stress
     * \param [out] aResult - yield stress
     * \param [in]  aLocalState
     * \param [in]  aParameters
    **********************************************************************************/
    void
    operator()(Plato::ScalarMultiVectorT< ResultT     > const& aResult,
               Plato::ScalarMultiVectorT< LocalStateT > const& aLocalState,
               Kokkos::View< Plato::ScalarVectorT< ControlT > *,
                             Kokkos::CudaUVMSpace > const& aParameters) const override
  {
      // Method used with the factory and has it own Kokkos parallel_for
      const Plato::OrdinalType tNumCells = aResult.extent(0);
      const Plato::OrdinalType tNumTerms = aResult.extent(1);

      // Strings for mapping parameter names to the equation
      // variables. Note: the LocalState is a required parameter
      // though possibly not used. That is in the operator() just the
      // parameters after the aLocalState parameter are optional.
      std::vector< std::string > tParamLabels( aParameters.extent(0) );

      for( Plato::OrdinalType i=0; i<tParamLabels.size(); ++i )
      {
        tParamLabels[i] = aParameters(i).label();
      }

      // The local state label is always last.
      tParamLabels.push_back( aLocalState.label() );

      const Plato::OrdinalType tNumParamLabels = tParamLabels.size();

      // If the user wants to use the input parameters these hold the
      // names of the equation variables that are mapped to the input
      // parameters.
      Kokkos::View< VariableMap *, Kokkos::CudaUVMSpace >
        tVarMaps ("Yield Stress Exp. Variable Maps", tNumParamLabels);

      // No mappings initially.
      for( Plato::OrdinalType i=0; i<tNumParamLabels; ++i )
        tVarMaps(i).key = 0;

      /*!< expression evaluator */
      ExpressionEvaluator< Plato::ScalarMultiVectorT< ResultT >,
                           Plato::ScalarMultiVectorT< LocalStateT >,
                           Plato::ScalarVectorT< ControlT >,
                           ControlT > tExpEval;

      // Look for a Custom Plasticity Model
      if( mInputParams.isSublist("Custom Plasticity Model") )
      {
        auto tCPMParams = mInputParams.sublist("Custom Plasticity Model");

        // Get the expression from the parameters.
        std::string tEquationStr = ParseTools::getEquationParam(tCPMParams);

        // Parse the expression.
        tExpEval.parse_expression(tEquationStr.c_str());

        // For all of the variables found in the expression optionally
        // get their values from the parameter list.
        const std::vector< std::string > tVarNames =
          tExpEval.get_variables();

        for( auto const & tVarName : tVarNames )
        {
          // Here the expression variable is found as a Plato::Scalar
          // so the value comes from the XML and is set directly.
          if( tCPMParams.isType<Plato::Scalar>(tVarName) )
          {
            // The value *MUST BE* converted to the ControlT as it is
            // the type used for all fixed variables.
            ControlT tVal = tCPMParams.get<Plato::Scalar>(tVarName);

            tExpEval.set_variable( tVarName.c_str(), tVal );
          }
          // Here the expression variable is found as a string so the
          // values should come from the XML.
          else if( tCPMParams.isType<std::string>(tVarName) )
          {
            std::string tVal = tCPMParams.get<std::string>(tVarName);

            // These are the names of the parameters passed into the
            // evaluation operator below. If the equation variable
            // "value" matches then the parameter value will be used.
            bool tFound = false;

            for( Plato::OrdinalType i=0; i<tNumParamLabels; ++i )
            {
              if( tVal == tParamLabels[i] )
              {
                tVarMaps(i).key = 1;
                strcpy( tVarMaps(i).value, tVarName.c_str() );

                tFound = true;
                break;
              }
            }

            if( !tFound )
            {
              std::stringstream errorMsg;
              errorMsg << "Invalid parameter name '" << tVal << "' "
                       << "found for varaible name '" << tVarName << "'. "
                       << "It must be :";

              for( Plato::OrdinalType i=0; i<tNumParamLabels; i++ )
              {
                errorMsg << " '" << tParamLabels[i] << "'";
              }

              errorMsg << ".";

             THROWERR(  errorMsg.str() );
            }
          }
          // Here the expression variable should come from the
          // parameters passed in.
          else
          {
            // These are the names of the parameters passed into the
            // evaluation operator below. If the equation variable
            // name matches then the parameter value will be used.
            bool tFound = false;

            for( Plato::OrdinalType i=0; i<tNumParamLabels; ++i )
            {
              if( tVarName == tParamLabels[i] )
              {
                tVarMaps(i).key = 1;
                strcpy( tVarMaps(i).value, tVarName.c_str() );

                tFound = true;
                break;
              }
            }

            if( !tFound )
            {
              std::stringstream errorMsg;
              errorMsg << "Invalid varaible name '" << tVarName << "'. "
                       << "It must be :";

              for( Plato::OrdinalType i=0; i<tNumParamLabels; i++ )
              {
                errorMsg << " '" << tParamLabels[i] << "'";
              }

              errorMsg << ".";

             THROWERR(  errorMsg.str() );
            }
          }
        }
      }

      // If for some reason the expression evalutor is called but
      // without the XML block.
      else
      {
        THROWERR("Warning: Failed to find a 'Custom Plasticity Model' block.");
      }

      // After the parsing, set up the storage. The sizes must match
      // the input and output data sizes.
      tExpEval.setup_storage( tNumCells, tNumTerms );

      // The LocalState is a two-dimensional array with the first
      // indice being the cell index.
      if( tVarMaps(tNumParamLabels-1).key )
        tExpEval.set_variable( tVarMaps(tNumParamLabels-1).value, aLocalState );

      // Finally do the evaluation.
      Kokkos::parallel_for("Compute yield stress",
                           Kokkos::RangePolicy<>(0, tNumCells),
                           LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
      {
        // Values that change based on the aCellOrdinal index. These
        // are values that the user has requested to come from the
        // input parameters. The last is the LocalState and is handled
        // above, thus the reason for subtracting one.
        for( Plato::OrdinalType i=0; i<tNumParamLabels-1; ++i )
        {
          if( tVarMaps(i).key )
            tExpEval.set_variable( tVarMaps(i).value,
                                   (aParameters(i))(aCellOrdinal),
                                   aCellOrdinal );
        }

        // Evaluate the expression for this cell.
        tExpEval.evaluate_expression( aCellOrdinal, aResult );
      } );

      // Because there are views of views are used locally which are
      // reference counted and deleting the parent view DOES NOT
      // de-reference the child views a dummy view with no memory is
      // used to replace the child so it is de-referenced.  There is
      // still a slight memory leak because the creation of the dummy
      // views.
      Plato::ScalarVectorT< ControlT > tDummyVector
        ( "Yield Stress Exp. Dummy Parameter" );

      // Drop all of the references to the parameter data.
      for( Plato::OrdinalType i=0; i<aParameters.extent(0); ++i )
      {
        aParameters(i) = tDummyVector;
      }

      // Clear the temporary storage used in the expression
      // otherwise there will be memory leaks.
      tExpEval.clear_storage();
  }
};
// class YieldStressExpression

}// namespace Plato
#endif

#ifdef PLATOANALYZE_2D
PLATO_EXPL_DEC_INC_LOCAL_1(Plato::YieldStressExpression, Plato::SimplexPlasticity,       2)
PLATO_EXPL_DEC_INC_LOCAL_1(Plato::YieldStressExpression, Plato::SimplexThermoPlasticity, 2)
#endif

#ifdef PLATOANALYZE_3D
PLATO_EXPL_DEC_INC_LOCAL_1(Plato::YieldStressExpression, Plato::SimplexPlasticity,       3)
PLATO_EXPL_DEC_INC_LOCAL_1(Plato::YieldStressExpression, Plato::SimplexThermoPlasticity, 3)
#endif
