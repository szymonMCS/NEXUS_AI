// pages/SignInPage.tsx
/**
 * Sign In Page - Clerk authentication
 */

import { SignIn } from '@clerk/clerk-react';
import { Zap } from 'lucide-react';
import { Link } from 'react-router-dom';

export function SignInPage() {
  return (
    <div className="min-h-screen bg-background flex flex-col items-center justify-center p-4">
      {/* Logo */}
      <Link to="/" className="flex items-center gap-2 mb-8 group">
        <div className="w-12 h-12 rounded-xl bg-gradient-primary flex items-center justify-center glow-primary-sm group-hover:scale-110 transition-transform duration-300">
          <Zap className="w-7 h-7 text-white" />
        </div>
        <span className="text-2xl font-bold text-white">
          Sport<span className="text-gradient">AI</span>
        </span>
      </Link>

      {/* Clerk Sign In Component */}
      <SignIn
        appearance={{
          elements: {
            rootBox: 'mx-auto',
            card: 'bg-gray-900 border border-white/10 shadow-xl',
            headerTitle: 'text-white',
            headerSubtitle: 'text-gray-400',
            socialButtonsBlockButton: 'bg-white/5 border-white/10 text-white hover:bg-white/10',
            socialButtonsBlockButtonText: 'text-white',
            dividerLine: 'bg-white/10',
            dividerText: 'text-gray-400',
            formFieldLabel: 'text-gray-300',
            formFieldInput: 'bg-white/5 border-white/10 text-white placeholder:text-gray-500',
            formButtonPrimary: 'bg-gradient-to-r from-violet-500 to-purple-600 hover:opacity-90',
            footerActionLink: 'text-violet-400 hover:text-violet-300',
            identityPreviewText: 'text-white',
            identityPreviewEditButton: 'text-violet-400',
            formFieldAction: 'text-violet-400',
            formFieldInputShowPasswordButton: 'text-gray-400',
            otpCodeFieldInput: 'bg-white/5 border-white/10 text-white',
            formResendCodeLink: 'text-violet-400',
            alertText: 'text-gray-300',
            footer: 'hidden',
          },
        }}
        routing="path"
        path="/sign-in"
        signUpUrl="/sign-up"
        afterSignInUrl="/dashboard"
      />

      {/* Footer */}
      <p className="mt-8 text-sm text-gray-500">
        Nie masz konta?{' '}
        <Link to="/sign-up" className="text-violet-400 hover:text-violet-300">
          Zarejestruj się
        </Link>
      </p>
    </div>
  );
}

export default SignInPage;
