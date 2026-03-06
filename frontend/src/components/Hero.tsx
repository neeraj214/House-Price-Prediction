"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import { Calculator, ArrowRight, Home, Zap, MapPin } from "lucide-react";

export default function Hero() {
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.15,
      },
    },
  };

  const itemVariants = {
    hidden: { y: 30, opacity: 0 },
    visible: { y: 0, opacity: 1, transition: { duration: 0.6, ease: "easeOut" } },
  };

  return (
    <section className="relative overflow-hidden pt-32 pb-24 lg:pt-48 lg:pb-32 bg-gradient-to-br from-slate-50 to-blue-50/50">
      <div className="absolute top-0 left-0 right-0 h-px bg-slate-200" />
      <div className="absolute top-[-20%] left-[-10%] w-[50%] h-[50%] bg-blue-100/30 blur-[120px] rounded-full pointer-events-none" />
      <div className="absolute bottom-[-20%] right-[-10%] w-[50%] h-[50%] bg-indigo-100/20 blur-[120px] rounded-full pointer-events-none" />

      <div className="container mx-auto px-6 lg:px-12 max-w-7xl relative z-10">
        <motion.div
          variants={containerVariants}
          initial="hidden"
          animate="visible"
          className="max-w-4xl mx-auto text-center"
        >
          <motion.div
            variants={itemVariants}
            className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-blue-100 text-blue-600 text-xs font-bold uppercase tracking-widest mb-8 border border-blue-200 shadow-sm"
          >
            <Zap className="w-4 h-4" />
            <span>AI-Powered Insights</span>
          </motion.div>

          <motion.h1
            variants={itemVariants}
            className="text-5xl lg:text-7xl font-extrabold text-slate-900 leading-tight mb-6"
          >
            Predict Your <span className="text-blue-600">Property Value</span> with AI
          </motion.h1>

          <motion.p
            variants={itemVariants}
            className="text-lg lg:text-xl text-slate-600 mb-12 max-w-2xl mx-auto leading-relaxed"
          >
            Get instant, data-driven estimates for your Indian real estate properties using our professional machine learning models. Accurate, reliable, and completely free.
          </motion.p>

          <motion.div variants={itemVariants} className="flex flex-col sm:flex-row items-center justify-center gap-6">
            <Link
              href="/predict"
              className="group flex items-center gap-3 px-8 py-4 bg-blue-600 text-white rounded-2xl font-bold text-lg hover:bg-blue-700 transition-all hover:shadow-xl hover:shadow-blue-200 active:scale-[0.98] w-full sm:w-auto justify-center"
            >
              <Calculator className="w-5 h-5" />
              <span>Start Prediction</span>
              <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
            </Link>
            <div className="flex -space-x-3 overflow-hidden">
              {[1, 2, 3, 4].map((i) => (
                <div
                  key={i}
                  className="inline-block h-10 w-10 rounded-full ring-2 ring-white bg-slate-200 border border-slate-300 overflow-hidden"
                >
                  <img
                    className="h-full w-full object-cover grayscale opacity-60"
                    src={`https://i.pravatar.cc/150?u=${i}`}
                    alt=""
                  />
                </div>
              ))}
              <div className="flex items-center pl-6 text-sm font-medium text-slate-500 italic">
                Trusted by 5,000+ users
              </div>
            </div>
          </motion.div>
        </motion.div>

        <motion.div
          variants={itemVariants}
          initial="hidden"
          animate="visible"
          className="mt-24 grid grid-cols-1 md:grid-cols-3 gap-8"
        >
          <FeatureCard
            icon={<Zap className="w-6 h-6 text-blue-600" />}
            title="Instant Predictions"
            description="Get results in milliseconds with our high-performance heuristic models."
          />
          <FeatureCard
            icon={<Home className="w-6 h-6 text-blue-600" />}
            title="Accurate ML Model"
            description="Trained on thousands of property transactions for maximum precision."
          />
          <FeatureCard
            icon={<MapPin className="w-6 h-6 text-blue-600" />}
            title="Location-Based Pricing"
            description="Accounts for regional pricing differences across major Indian metros."
          />
        </motion.div>
      </div>
    </section>
  );
}

function FeatureCard({ icon, title, description }: { icon: React.ReactNode; title: string; description: string }) {
  return (
    <div className="bg-white/80 backdrop-blur-sm p-8 rounded-3xl border border-slate-100 shadow-sm hover:shadow-md transition-shadow group">
      <div className="w-14 h-14 bg-blue-50 rounded-2xl flex items-center justify-center mb-6 group-hover:scale-110 transition-transform">
        {icon}
      </div>
      <h3 className="text-xl font-bold text-slate-900 mb-3">{title}</h3>
      <p className="text-slate-500 leading-relaxed">{description}</p>
    </div>
  );
}
