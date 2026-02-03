import { SignIn } from '@clerk/clerk-react';
import { BarChart3 } from 'lucide-react';

export function SignInPage() {
  return (
    <div className="min-h-screen stadium-bg flex items-center justify-center p-4">
      <div className="w-full max-w-md">
        {/* Logo */}
        <div className="text-center mb-8">
          <div className="w-16 h-16 mx-auto mb-4 rounded-2xl bg-gradient-to-br from-[#00d4ff] to-[#00ff88] flex items-center justify-center shadow-lg shadow-[#00d4ff]/30">
            <BarChart3 className="w-8 h-8 text-black" />
          </div>
          <h1 className="text-2xl font-bold text-white">Welcome Back</h1>
          <p className="text-gray-400 mt-2">Sign in to access your predictions</p>
        </div>

        {/* Clerk SignIn with custom styling */}
        <div className="bg-[#141b2d] border border-white/10 rounded-2xl p-1">
          <SignIn 
            routing="path" 
            path="/sign-in"
            signUpUrl="/sign-up"
            redirectUrl="/dashboard"
            appearance={{
              elements: {
                rootBox: 'w-full',
                card: 'bg-transparent shadow-none',
                headerTitle: 'text-white text-xl',
                headerSubtitle: 'text-gray-400',
                socialButtonsBlockButton: 'bg-white/5 border-white/10 text-white hover:bg-white/10',
                formFieldLabel: 'text-gray-300',
                formFieldInput: 'bg-white/5 border-white/10 text-white focus:border-[#00d4ff]',
                formButtonPrimary: 'bg-gradient-to-r from-[#00d4ff] to-[#00ff88] text-black font-bold hover:opacity-90',
                footerActionText: 'text-gray-400',
                footerActionLink: 'text-[#00d4ff] hover:text-[#00ff88]',
                dividerLine: 'bg-white/10',
                dividerText: 'text-gray-500',
                identityPreviewText: 'text-white',
                identityPreviewEditButton: 'text-[#00d4ff]',
              }
            }}
          />
        </div>

        {/* Back to home */}
        <div className="text-center mt-6">
          <a href="/" className="text-gray-500 hover:text-white transition-colors text-sm">
            ← Back to home
          </a>
        </div>
      </div>
    </div>
  );
}

export default SignInPage;
