#include <iostream>
#include <cmath>
#include <vector>
#include <functional>

#define matrix std::vector<std::vector<double>>
#define vec std::vector<double>

struct Context
{
	std::function<double(double,double)> u_exact;
	double A;
	double T;

	std::function<double(double,double)> a;
	std::function<double(double,double)> b;
	std::function<double(double,double)> c;
	std::function<double(double,double)> f;
	std::function<double(double)> alpha;
	std::function<double(double)> beta;
	std::function<double(double)> phi;
};



vec L(
	double h,
	int k,
	const vec& x,
	const vec& t,
	const matrix& u,
	const Context& context
)
{
	vec Lu(u.size()); 

	// i \in 1...N-1
	for (size_t i=1; i<u.size()-1; ++i)
	{
		double uxx = (u[i+1][k] - 2 * u[i][k] + u[i-1][k]) / (h * h);

		double ux = (u[i+1][k] - u[i-1][k]) / (2 * h);

		double u0 = u[i][k];
		
		Lu[i] = 
			context.a(x[i], t[k]) * uxx +
			context.b(x[i], t[k]) * ux  +
			context.c(x[i], t[k]) * u0;
	}

	return Lu;
}


matrix explicit_scheme(
	int N,
	int M,
	double T,
	const Context& context
)
{
	double h = 1.0 / N;
	double tau = T / M;
	
	vec x(N + 1);
	for (int i=0; i<=N; ++i)
		x[i] = i * h;

	vec t(M + 1);
	for (int k=0; k<=M; ++k)
		t[k] = k * tau;

	// u[i][k]
	matrix u(N + 1, std::vector<double>(M + 1));
	for (int i=0; i<=N; ++i)
		u[i][0] = context.phi(i * h);

	for (int k=1; k<=M; ++k)
	{
		vec Lu = L(h, k - 1, x, t, u, context);
		
		for (int i=1; i<=N-1; ++i)
		{
			u[i][k] = u[i][k-1] + tau * (Lu[i] + context.f(x[i], t[k-1]));
			
			u[0][k] = (-1.0 / 3) * (2 * h * context.alpha(t[k]) + u[2][k] - 4 * u[1][k] );

			u[N][k] = context.beta(t[k]);
		}
	}

	return u;
}


vec progonka(
	const vec& A,
	const vec& B,
	const vec& C,
	const vec& G,
	int N
)
{
	vec s(N+1);
	vec t(N+1);
	
	s[0] = C[0] / B[0];
	t[0] = -G[0] / B[0];
	for (int i=1; i<=N-1; ++i)
	{
		s[i] = C[i] / (B[i] - A[i] * s[i - 1]);
		t[i] = (G[i] - A[i] * t[i-1]) / (A[i] * s[i-1] - B[i]);
	}
	s[N] = 0;
	t[N] = (G[N] - A[N] * t[N-1]) / (A[N] * s[N-1] - B[N]);

	vec solution(N + 1);
	solution[N] = t[N];
	for (int i=N-1; i>=0; --i)
	{
		solution[i] = s[i] * solution[i + 1] + t[i];
	}

	return solution;
}


matrix implicit_scheme(
	int N,
	int M,
	double T,
	const Context& context,
	double sigma,
	// new_tk(t,k,tau)
	std::function<double(vec, int, double)> new_tk
)
{
	
	double h = 1.0 / N;
	double tau = T / M;
	
	vec x(N + 1);
	for (int i=0; i<=N; ++i)
		x[i] = i * h;

	vec t(M + 1);
	for (int k=0; k<=M; ++k)
		t[k] = k * tau;

	// u[i][k]
	matrix u(N + 1, std::vector<double>(M + 1));
	for (int i=0; i<=N; ++i)
		u[i][0] = context.phi(i * h);

	for (int k=1; k<=M; ++k)
	{
		vec G(N + 1);
		vec A(N + 1);
		vec B(N + 1);
		vec C(N + 1);
		
		B[0] = 1 / h;
		C[0] = 1 / h;

		A[N] = 0.0;
		B[N] = -1;

		for (int i=1; i <= N-1; ++i)
		{
			A[i] = sigma * (
						context.a(x[i],t[k]) / (h * h)
						-context.b(x[i],t[k]) / (2 * h)
					);

			B[i] = sigma * (
						2 * context.a(x[i],t[k]) / (h * h)
						- context.c(x[i],t[k])
						+ 1 / (tau * sigma)
					);

			C[i] = sigma * (
						context.a(x[i], t[k]) / (h * h)
						+ context.b(x[i], t[k]) / (2 * h)
					);
		}

		G[0] = context.alpha(t[k]);
		G[N] = context.beta(t[k]);
		for (int i=1; i <= N-1; ++i)
		{
			vec Lu = L(h, k-1, x, t, u, context);
			G[i] = 
				- (1 / tau) * u[i][k-1] 
				- (1 - sigma) * Lu[i]
				- context.f(x[i], new_tk(t, k, tau));
		}
		

		vec solution = progonka(A, B, C, G, N);
		
		for (int i=0; i<=N; ++i)
			u[i][k] = solution[i];
	}

	return u;
}

int get_opt_m(
	int N,
	double T,
	double A
)
{
	int M = 5;
	double h_sq = 1.0 / (N * N);

	for (int i=0; i<100; ++i)
	{
		double tau = T / M;
		if (A * tau / (h_sq) <= 0.5) break;
		M *= 2;
	}

	return M;
}


void helper_big_grid(
	int N,
	int M,
	const vec& t,
	const vec& x,
	const matrix& u
)
{

	printf("          ");
	for(int i=0; i<6; ++i)
	{
		int x_ix = i * (N / 5.0);
		printf("x=%5.3f ", x[x_ix]);
	}
	printf("\n");

	//printing here...
	for (int k=0; k<6; ++k)
	{
		int t_ix = k * (M / 5);
		printf("t=%.3f  ", t[t_ix]);

		for (int i=0; i<6; ++i)
		{ 
			int x_ix = i * (N / 5.0);
			printf("%8.4f", u[x_ix][t_ix]);
		}
		printf("\n");
	}
	printf("\n\n");
}

void print_big_grid(
	const Context& context
)
{
	double T = context.T;
	int N = 5;
	
	for (int j=0; j<3; ++j)
	{
		printf("for N=%d\n", N);
		int M = get_opt_m(N, T, context.A);
		double h = 1.0 / N;
		double tau = T / M;
		matrix u = explicit_scheme(N, M, T, context);
		matrix u05 = implicit_scheme(
						N, M, T, context, 0.5,
						[](vec _t, int _k, double _tau){
							return _t[_k] - _tau / 2;
						}
					);
		matrix u1 = implicit_scheme(
						N, M, T, context, 1,
						[](vec _t, int _k, double _tau){
							return _t[_k];
						}
					);
		
		vec x(N + 1);
		for (int i=0; i<=N; ++i)
			x[i] = i * h;

		vec t(M + 1);
		for (int k=0; k<=M; ++k)
			t[k] = k * tau;
		
		printf("explicit\n");
		helper_big_grid(N, M, t, x, u);
		
		printf("implicit sigma=0.5\n");
		helper_big_grid(N, M, t, x, u05);

		printf("implicit sigma=1\n");
		helper_big_grid(N, M, t, x, u1);
		

		N *= 2;
	}
	
}


void helper_residuals(
	int N,
	int M,
	int N1,
	int M1,
	double tau,
	double h,
	const matrix& u,
	const matrix& u1,
	const Context & context
)
{

	double res = -1;
	for (int i=0; i<=N; ++i)
	{
		for (int k=0; k<=M; ++k)
		{
			double u_ex = context.u_exact(
					i * h,
					k * tau
			);
			double d = fabs(u[i][k] - u_ex); 
			res = std::max(res, d);
		}
	}

	double res1 = -1;
	for (int k=0; k<6; ++k)
	{
		int t_ix = k * (M / 5);
		int t_ix1 = k * (M1 / 5);
		for (int i=0; i<6; ++i)
		{ 
			int x_ix = i * (N / 5.0);
			int x_ix1 = i * (N1 / 5.0);
			double d = fabs(u[x_ix][t_ix] - u1[x_ix1][t_ix1]);
			res1 = std::max(res1, d);
		}
	}

	printf("h=%.3e ", h);
	printf("tau=%.3e ", tau);
	printf("max_diff=%.3e ", res);
	printf("inner_diff=%.3e\n", res1);
}

void print_residuals(
	const Context& context 
)
{
	printf("explicit:\n");
	int N = 10;
	for (int j=0; j<5; ++j)
	{

		int M = get_opt_m(N, context.T, context.A);
		double tau = context.T / M;
		double h = 1.0 / N;
		
		matrix u = explicit_scheme(N, M, context.T, context);
		
		int N1 = N / 2;
		int M1 = get_opt_m(N1, context.T, context.A);
		matrix u1 = explicit_scheme(N1, M1, context.T, context);
		
		helper_residuals(N, M, N1, M1, tau, h, u, u1, context);
		
		N *= 2;
	}

	int M = 100;
	printf("\n\nM_implicit=%d\n", M);
	printf("implicit sigma=0.5:\n");
	N = 10;
	for (int j=0; j<5; ++j)
	{

		double tau = context.T / M;
		double h = 1.0 / N;
		
		int N1 = N / 2;
		
		matrix u05 = implicit_scheme(
						N, M, context.T, context, 0.5,
						[](vec _t, int _k, double _tau){
							return _t[_k] - _tau / 2;
						}
					);
		matrix u05_1 = implicit_scheme(
						N1, M, context.T, context, 0.5,
						[](vec _t, int _k, double _tau){
							return _t[_k] - _tau / 2;
						}
					);

		helper_residuals(
					N, M, N1, M, tau,
					h, u05, u05_1, context
				);

		N *= 2;
	}

	printf("\nimplicit sigma=1:\n");
	N = 10;
	for (int j=0; j<5; ++j)
	{

		double tau = context.T / M;
		double h = 1.0 / N;
		
		int N1 = N / 2;
		
		matrix u1 = implicit_scheme(
						N, M, context.T, context, 1,
						[](vec _t, int _k, double _tau){
							return _t[_k];
						}
					);
		matrix u1_1 = implicit_scheme(
						N1, M, context.T, context, 1,
						[](vec _t, int _k, double _tau){
							return _t[_k];
						}
					);

		helper_residuals(
					N, M, N1, M, tau,
					h, u1, u1_1, context
				);

		N *= 2;
	}

}

void ex0()
{
	Context cntx;

	cntx.T = 0.1;
	auto u_exact = [](double x, double t){
		return x + t;
	};
	cntx.u_exact = u_exact;

	cntx.A = 1.0;

	cntx.a = [](double x, double t){
		return cos(x);
	};

	cntx.b = [](double x, double t){
		// b==0
		return 0;
	};

	cntx.c = [](double x, double t){
		// c==0
		return 0;
	};
	
	cntx.f = [](double x, double t){
		return 1;
	};

	cntx.alpha = [](double t){
		return 1;
	};
	
	cntx.beta = [&](double t){
		return u_exact(1, t);
	};

	cntx.phi = [&](double x){
		return u_exact(x, 0);
	};

	print_big_grid(cntx);

	print_residuals(cntx);
}


void ex1()
{
	Context cntx;

	cntx.T = 0.1;
	auto u_exact = [](double x, double t){
		return x*x*x + t*t*t;
	};
	cntx.u_exact = u_exact;

	cntx.A = 1.0;

	cntx.a = [](double x, double t){
		return cos(x);
	};

	cntx.b = [](double x, double t){
		// b==0
		return 0;
	};

	cntx.c = [](double x, double t){
		// c==0
		return 0;
	};
	
	cntx.f = [](double x, double t){
		return 3*t*t - cos(x)*6*x;
	};

	cntx.alpha = [](double t){
		return 0;
	};
	
	cntx.beta = [&](double t){
		return u_exact(1, t);
	};

	cntx.phi = [&](double x){
		return u_exact(x, 0);
	};

	print_big_grid(cntx);

	print_residuals(cntx);
}


void ex2()
{
	Context cntx;

	cntx.T = 0.1;
	auto u_exact = [](double x, double t){
		return sin(2 * t + 1) + cos(2 * x);
	};
	cntx.u_exact = u_exact;

	cntx.A = 1.0;

	cntx.a = [](double x, double t){
		return cos(x);
	};

	cntx.b = [](double x, double t){
		// b==0
		return 0;
	};

	cntx.c = [](double x, double t){
		// c==0
		return 0;
	};
	
	cntx.f = [](double x, double t){
		return 2*cos(2*t+1) - cos(x) * (-4*cos(2*x));
	};

	cntx.alpha = [](double t){
		return 0;
	};
	
	cntx.beta = [&](double t){
		return u_exact(1, t);
	};

	cntx.phi = [&](double x){
		return u_exact(x, 0);
	};

	print_big_grid(cntx);

	print_residuals(cntx);
}

int main()
{
	printf("ex0: u = x + t\n");
	ex0();

	printf("\n\n\n\nex1: u = x^3 + t^3\n");
	ex1();

	printf("\n\n\n\nex2: u = sin(2t+1) + cos(2x)\n");
	ex2();
}


