#ifndef CUSTOMMATERIAL_HPP
#define CUSTOMMATERIAL_HPP

#include "PlatoTypes.hpp"

#include "alg/PlatoLambda.hpp"

#include <Teuchos_ParameterList.hpp>

#include "ExpressionEvaluator.hpp"

#include <Kokkos_Parallel.hpp>

#include <Sacado.hpp>

#include <cstdarg>

#define DO_KOKKOS 1

namespace Plato
{

/******************************************************************************/
/*!
  \brief Class for custom material models
*/
class CustomMaterial
/******************************************************************************/
{
public:
    CustomMaterial(const Teuchos::ParameterList& aParamList) {}
    virtual ~CustomMaterial() {}

//protected:
    virtual Plato::Scalar GetCustomExpressionValue(
        const Teuchos::ParameterList& paramList,
        const std::string equationName ) const;

    virtual Plato::Scalar GetCustomExpressionValue(
        const Teuchos::ParameterList& paramList,
        const Plato::OrdinalType equationIndex = (Plato::OrdinalType) -1,
        const std::string equationName = std::string("Equation") ) const;

    void getTypedValue( const Plato::Scalar val,
                        const size_t ith, const size_t size,
                        Plato::Scalar &value ) const {
      value = val;
    };

    void getTypedValue( const Plato::Scalar val,
                        const size_t ith, const size_t size,
                        Sacado::Fad::DFad<Plato::Scalar> &value ) const {
      value = Sacado::Fad::DFad<Plato::Scalar>(size, ith, val);
    };

    template< typename T >
    void getTypedValue( const Plato::Scalar val,
                        const size_t ith, const size_t size,
                        T &value ) const {
      THROWERR( "Unknown type value requested." );
    };

    KOKKOS_INLINE_FUNCTION
    void localPrintf( Plato::Scalar val ) const
    {
      printf( "%f ", val );
    }

    template< typename T >
    KOKKOS_INLINE_FUNCTION
    void localPrintf( T val ) const
    {
      printf( "%f %f ", val.val(), val.dx(0) );
    }

    // Template method which does the real work.
    template< typename TYPE >
    void
    GetCustomExpressionValue( const Teuchos::ParameterList& paramList,
                              const std::string equationStr,
                                    TYPE & result ) const
    {
      // This code is what should normally be executed. It is Host (CPU) only.
      if( 0 )
      {
        const size_t nThreads = 1;
        const size_t nValues = 1;

        // Create an expression evaluator.
        ExpressionEvaluator< Kokkos::View< TYPE *              , Kokkos::HostSpace>,
                             Kokkos::View< TYPE *              , Kokkos::HostSpace>,
                             Kokkos::View< Plato::OrdinalType *, Kokkos::HostSpace>,
                             Plato::Scalar > expEval;

        // Parse the equation. The expression tree is held internally.
        expEval.parse_expression( equationStr.c_str() );

        // Set up the storage.
        expEval.setup_storage( nThreads, nValues );

        // For all of the variables found in the expression get
        // their values from the parameter list.
        const std::vector< std::string > variableNames =
          expEval.get_variables();

        Kokkos::View<TYPE*, Kokkos::HostSpace> inputs[variableNames.size()];

        size_t i = 0;
        for( auto const & variable : variableNames )
        {
          // Note the "2" is DFAD types so to get the right amount of
          // memory allocated. For scalars it is moot.
          inputs[i] = Kokkos::View<TYPE*, Kokkos::HostSpace>(variable, nValues, 2);

          // Type all of the input values.
          TYPE val;
          getTypedValue( paramList.get<Plato::Scalar>( variable ),
                         i, variableNames.size(), val );

          inputs[i](0) = val;

          expEval.set_variable( variable.c_str(), inputs[i] );

          i++;
        }

        // If a valid equation, evaluate it,
        if( expEval.valid_expression( true ) )
        {
          Kokkos::View<TYPE *, Kokkos::HostSpace> results("results", nValues, 2);
          expEval.evaluate_expression( 0, results );

          result = results(0);
        }

	// Clear the temporary storage used in the expression
	// otherwise there will be memory leaks.
        expEval.clear_storage();
      }

      // Example with one thread and one value. This code is CPU or
      // GPU as it uses UVM spaces.
      else if( 0 )
      {
        const size_t nThreads = 1;
        const size_t nValues = 1;

        // Create an expression evaluator and pass the parameter list so
        // variable values can be retrived.
        ExpressionEvaluator< Kokkos::View< TYPE *              , Kokkos::CudaUVMSpace>,
                             Kokkos::View< TYPE *              , Kokkos::CudaUVMSpace>,
                             Kokkos::View< Plato::OrdinalType *, Kokkos::CudaUVMSpace>,
                             Plato::Scalar > expEval;

        expEval.parse_expression( "E/((1.0+v)(1.0-2.0*v))" );

        // Set up the storage.
        expEval.setup_storage( nThreads, nValues );

        // For all of the variables found in the expression and get
        // their values from the parameter list.
        const std::vector< std::string > variableNames =
          expEval.get_variables();

        Plato::Scalar v = paramList.get<Plato::Scalar>( "v" );
        expEval.set_variable( "v", v, 0 );

        // Setup memory for the input data.
        Kokkos::View<TYPE *, Kokkos::CudaUVMSpace> E("E", nValues, 2);

        TYPE e;
        getTypedValue( paramList.get<Plato::Scalar>( "E" ), 0, 1, e );
        E[0] = e;

        expEval.set_variable( "E", E );

        std::cout << "________________________________" << std::endl
                  << "expression : " << equationStr << std::endl;

        std::cout << "________________________________" << std::endl;
        expEval.print_expression( std::cout );

        // If a valid equation, evaluate it,
        if( expEval.valid_expression( true ) )
        {
          std::cout << "________________________________" << std::endl;
          expEval.print_variables( std::cout );

          Kokkos::View<TYPE *, Kokkos::CudaUVMSpace> results("results", nValues, 2);

#ifdef DO_KOKKOS
          // Device - GPU
          Kokkos::parallel_for(Kokkos::RangePolicy<>(0, nThreads),
                               LAMBDA_EXPRESSION(const Plato::OrdinalType & tCellOrdinal)
                               {
                                 expEval.evaluate_expression( tCellOrdinal, results );
                               }, "Compute");

          // Wait for the GPU to finish so to get the data on to the CPU.
          Kokkos::fence();
#else
          // Host - CPU
          for( size_t tCellOrdinal=0; tCellOrdinal<nThreads; ++tCellOrdinal )
            expEval.evaluate_expression( tCellOrdinal, results );
#endif

          std::cout << "________________________________" << std::endl;
          for( size_t i=0; i<nValues; ++i )
          {
            std::cout << "results = ";

            for( size_t j=0; j<nThreads; ++j )
              std::cout << results(j,i) << "  ";

            std::cout << std::endl;
          }

          result = results[0];
        }

	// Clear the temporary storage used in the expression
	// otherwise there will be memory leaks.
        expEval.clear_storage();
      }
      // Example with two threads and ten values.  This code is CPU or
      // GPU as it uses UVM spaces.
      else if( 1 )
      {
        const size_t nThreads = 2;
        const size_t nValues = 10;

        // Create an expression evaluator and pass the parameter list so
        // variable values can be retrived.
        ExpressionEvaluator< Kokkos::View< TYPE **, Kokkos::CudaUVMSpace>,
                             Kokkos::View< TYPE **, Kokkos::CudaUVMSpace>,
                             Kokkos::View< Plato::Scalar *, Kokkos::CudaUVMSpace>,
                             Plato::Scalar > expEval;

        // Parse the equation. The expression tree is held internally.
        expEval.parse_expression( "E/((1.0+v)(1.0-2.0*v))" );

        // For all of the variables found in the expression and get
        // their values from the parameter list.
        const std::vector< std::string > variableNames =
          expEval.get_variables();

        expEval.setup_storage( nThreads, nValues );

        Plato::Scalar v = paramList.get<Plato::Scalar>( "v" );

	Kokkos::View< Plato::Scalar *, Kokkos::CudaUVMSpace> V ("V" , nValues);
	Kokkos::View< Plato::Scalar *, Kokkos::CudaUVMSpace> VV("VV", nValues);

        V[0] = VV[0] = v;
        for( size_t i=1; i<nValues; ++i )
        {
          V[i] = i * v;
          VV[i] = v;
        }

        expEval.set_variable( "v", V,  0 );
        if( nThreads == 2 )
          expEval.set_variable( "v", VV, 1 );

        Kokkos::View<TYPE **, Kokkos::CudaUVMSpace> E("E", nThreads, nValues, 2);

        TYPE e;
        getTypedValue( paramList.get<Plato::Scalar>( "E" ), 0, 1, e );
        E(0,0) = e;

        if( nThreads == 2 )
          E(1,0) = e;

        for( size_t i=1; i<nValues; ++i )
        {
          getTypedValue( i, 0, 1, e );
          E(0,i) = e;

          if( nThreads == 2 )
            E(1,i) = E(1,0);
        }

        expEval.set_variable( "E", E );

        std::cout << "________________________________" << std::endl
                  << "expression : " << equationStr << std::endl;

        std::cout << "________________________________" << std::endl;
        expEval.print_expression( std::cout );

        // If a valid equation, evaluate it,
        if( expEval.valid_expression( true ) )
        {
          std::cout << "________________________________" << std::endl;
          expEval.print_variables( std::cout );

          Kokkos::View<TYPE **, Kokkos::CudaUVMSpace> results("results", nThreads, nValues, 2);

#ifdef DO_KOKKOS
          // Device - GPU
          Kokkos::parallel_for(Kokkos::RangePolicy<>(0, nThreads),
                               LAMBDA_EXPRESSION(const Plato::OrdinalType & tCellOrdinal)
                               {
                                 expEval.evaluate_expression( tCellOrdinal, results );
                               }, "Compute");

          // Wait for the GPU to finish so to get the data on to the CPU.
          Kokkos::fence();
#else
          // Host - CPU
          for( size_t tCellOrdinal=0; tCellOrdinal<nThreads; ++tCellOrdinal )
            expEval.evaluate_expression( tCellOrdinal, results );
#endif

          std::cout << "________________________________" << std::endl;
          for( size_t i=0; i<nValues; ++i )
          {
            std::cout << "results = ";

            for( size_t j=0; j<nThreads; ++j )
              std::cout << results(j,i) << "  ";

            std::cout << std::endl;
          }

          result = results(0,0);
        }

	// Clear the temporary storage used in the expression
	// otherwise there will be memory leaks.
        expEval.clear_storage();
      }
    };
};
// class CustomMaterial

} // namespace Plato

#endif
