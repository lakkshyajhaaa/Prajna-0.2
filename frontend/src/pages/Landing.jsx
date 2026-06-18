import React from 'react';
import { Link } from 'react-router-dom';
import { ArrowRight, Fingerprint, Terminal, Crosshair, Cpu } from 'lucide-react';
import { motion } from 'framer-motion';
import Timeline from './Timeline';

const containerVariants = {
  hidden: { opacity: 0 },
  visible: { 
    opacity: 1,
    transition: { staggerChildren: 0.1, delayChildren: 0.2 }
  }
};

const itemVariants = {
  hidden: { opacity: 0, y: 20 },
  visible: { opacity: 1, y: 0, transition: { type: "spring", stiffness: 300, damping: 24 } }
};

const Landing = () => {
  return (
    <motion.div 
      className="hero-container"
      variants={containerVariants}
      initial="hidden"
      animate="visible"
    >
      <motion.div className="hero-pill" variants={itemVariants}>
        <span className="status-indicator status-green"></span> SYSTEM ONLINE // PRAJNA v0.2
      </motion.div>
      
      <motion.h1 className="hero-title" variants={itemVariants}>
        SMART<br/>
        IDENTITY<br/>
        VERIFICATION
      </motion.h1>
      
      <motion.p className="hero-subtitle" variants={itemVariants}>
        A privacy-first identity verification system. We use a smart, two-step process to securely recognize faces, ensuring blazing fast speeds while keeping your data 100% private.
      </motion.p>
      
      <motion.div className="btn-group" variants={itemVariants}>
        <Link to="/verify" className="btn btn-primary">
          VERIFY IDENTITY <ArrowRight size={18} />
        </Link>
        <Link to="/database" className="btn btn-secondary">
          MANAGE PROFILES <Fingerprint size={18} />
        </Link>
      </motion.div>
      
      <motion.div className="grid-3" style={{ width: '100%', marginTop: '2rem' }} variants={itemVariants}>
        <div>
          <Crosshair size={24} color="var(--accent)" style={{ marginBottom: '1.5rem' }} />
          <h3 style={{ fontSize: '1rem', marginBottom: '0.5rem' }}>SMART ROUTING</h3>
          <p className="mono-label" style={{ color: 'var(--text-secondary)', textTransform: 'none' }}>
            Skips heavy processing when it's absolutely sure. Only performs deep scans when it encounters difficult or blurry images.
          </p>
        </div>
        
        <div>
          <Terminal size={24} color="var(--accent)" style={{ marginBottom: '1.5rem' }} />
          <h3 style={{ fontSize: '1rem', marginBottom: '0.5rem' }}>RESPONSIBILITY SCORES</h3>
          <p className="mono-label" style={{ color: 'var(--text-secondary)', textTransform: 'none' }}>
            Mathematical certainty. Every decision is scored, audited, and mathematically bound before authorization.
          </p>
        </div>
        
        <div>
          <Cpu size={24} color="var(--accent)" style={{ marginBottom: '1.5rem' }} />
          <h3 style={{ fontSize: '1rem', marginBottom: '0.5rem' }}>TWO-STEP VERIFICATION</h3>
          <p className="mono-label" style={{ color: 'var(--text-secondary)', textTransform: 'none' }}>
            Quick Scan for extreme speed. Deep Scan for careful, detailed analysis. The perfect balance.
          </p>
        </div>
      </motion.div>
      
      <div style={{ marginTop: '6rem', width: '100%', borderTop: '1px solid var(--border-color)', paddingTop: '2rem' }}>
        <Timeline />
      </div>
    </motion.div>
  );
};

export default Landing;
