import React from 'react';
import { Link } from 'react-router-dom';

const Footer = () => {
  return (
    <footer style={{
      borderTop: '1px solid var(--border-color)',
      marginTop: 'auto',
      padding: '2rem 3rem',
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      fontFamily: 'var(--font-mono)',
      fontSize: '0.75rem',
      color: 'var(--text-secondary)'
    }}>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
        <span>Prajna Framework - Experimental Release</span>
        <span>
          Developed by <a href="https://lakkshyajha.me" target="_blank" rel="noopener noreferrer" style={{ color: 'var(--accent)', textDecoration: 'none' }}>Lakkshya Jha</a>
        </span>
      </div>
      <div style={{ display: 'flex', gap: '1.5rem' }}>
        <Link to="/privacy" style={{ color: 'var(--text-secondary)', textDecoration: 'none' }}>Privacy Policy</Link>
        <Link to="/terms" style={{ color: 'var(--text-secondary)', textDecoration: 'none' }}>Terms of Service</Link>
        <Link to="/disclaimer" style={{ color: 'var(--text-secondary)', textDecoration: 'none' }}>Disclaimer</Link>
      </div>
    </footer>
  );
};

export default Footer;
