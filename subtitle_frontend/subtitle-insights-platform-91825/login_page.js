import React, { useState } from 'react';

// PUBLIC_INTERFACE
/**
 * LoginPage Component
 * A modern login page with Ocean Professional theme
 * Features a centered card layout on a gradient background
 */
export default function LoginPage() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    console.log('Login submitted:', { email, password });
    // Placeholder for authentication logic
  };

  return (
    <div style={styles.container}>
      <div style={styles.card}>
        <div style={styles.header}>
          <h1 style={styles.title}>Welcome Back</h1>
          <p style={styles.subtitle}>Sign in to your account to continue</p>
        </div>

        <form onSubmit={handleSubmit} style={styles.form}>
          <div style={styles.inputGroup}>
            <label htmlFor="email" style={styles.label}>
              Email Address
            </label>
            <input
              id="email"
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="Enter your email"
              style={styles.input}
              required
            />
          </div>

          <div style={styles.inputGroup}>
            <label htmlFor="password" style={styles.label}>
              Password
            </label>
            <input
              id="password"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="Enter your password"
              style={styles.input}
              required
            />
          </div>

          <button type="submit" style={styles.button}>
            Sign In
          </button>
        </form>

        <div style={styles.footer}>
          <p style={styles.footerText}>
            Don't have an account?{' '}
            <a href="#signup" style={styles.link}>
              Sign up here
            </a>
          </p>
        </div>
      </div>
    </div>
  );
}

// Styles with Ocean Professional theme
const styles = {
  container: {
    minHeight: '100vh',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    background: 'linear-gradient(135deg, rgba(37, 99, 235, 0.1) 0%, #f9fafb 100%)',
    padding: '20px',
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
  },
  card: {
    backgroundColor: '#ffffff',
    borderRadius: '16px',
    boxShadow: '0 4px 6px rgba(0, 0, 0, 0.07), 0 10px 20px rgba(0, 0, 0, 0.05)',
    padding: '48px 40px',
    width: '100%',
    maxWidth: '440px',
    transition: 'transform 0.3s ease, box-shadow 0.3s ease',
  },
  header: {
    textAlign: 'center',
    marginBottom: '32px',
  },
  title: {
    fontSize: '28px',
    fontWeight: '700',
    color: '#111827',
    margin: '0 0 8px 0',
    letterSpacing: '-0.5px',
  },
  subtitle: {
    fontSize: '15px',
    color: '#6B7280',
    margin: '0',
    lineHeight: '1.5',
  },
  form: {
    display: 'flex',
    flexDirection: 'column',
    gap: '20px',
  },
  inputGroup: {
    display: 'flex',
    flexDirection: 'column',
    gap: '8px',
  },
  label: {
    fontSize: '14px',
    fontWeight: '500',
    color: '#374151',
    marginBottom: '4px',
  },
  input: {
    padding: '12px 16px',
    fontSize: '15px',
    border: '1.5px solid #E5E7EB',
    borderRadius: '8px',
    outline: 'none',
    transition: 'border-color 0.2s ease, box-shadow 0.2s ease',
    color: '#111827',
    backgroundColor: '#ffffff',
    fontFamily: 'inherit',
  },
  button: {
    marginTop: '8px',
    padding: '14px 24px',
    fontSize: '16px',
    fontWeight: '600',
    color: '#ffffff',
    backgroundColor: '#2563EB',
    border: 'none',
    borderRadius: '8px',
    cursor: 'pointer',
    transition: 'background-color 0.2s ease, transform 0.1s ease',
    boxShadow: '0 2px 4px rgba(37, 99, 235, 0.2)',
  },
  footer: {
    marginTop: '28px',
    textAlign: 'center',
    paddingTop: '24px',
    borderTop: '1px solid #E5E7EB',
  },
  footerText: {
    fontSize: '14px',
    color: '#6B7280',
    margin: '0',
  },
  link: {
    color: '#2563EB',
    textDecoration: 'none',
    fontWeight: '600',
    transition: 'color 0.2s ease',
  },
};

// Add hover and focus effects via style object
if (typeof document !== 'undefined') {
  const styleSheet = document.createElement('style');
  styleSheet.textContent = `
    input:focus {
      border-color: #2563EB !important;
      box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1) !important;
    }
    button:hover {
      background-color: #1D4ED8 !important;
      transform: translateY(-1px);
    }
    button:active {
      transform: translateY(0);
    }
    a:hover {
      color: #1D4ED8 !important;
      text-decoration: underline;
    }
  `;
  if (document.head) {
    document.head.appendChild(styleSheet);
  }
}
