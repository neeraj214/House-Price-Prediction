"use client";

import Navbar from "@/components/Navbar";
import Hero from "@/components/Hero";
import { motion } from "framer-motion";
import { Info, Mail, Github, Heart } from "lucide-react";

export default function Home() {
  return (
    <div className="flex flex-col min-h-screen bg-slate-50 selection:bg-blue-100 selection:text-blue-600">
      <Navbar />
      <main className="flex-grow">
        <Hero />
        
        {/* Why Choose Us Section */}
        <section className="py-24 bg-white">
          <div className="container mx-auto px-6 lg:px-12 max-w-7xl">
            <div className="text-center mb-20">
              <motion.h2
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                className="text-4xl lg:text-5xl font-bold text-slate-900 mb-6"
              >
                Accurate Indian <span className="text-blue-600">Real Estate</span> Insights
              </motion.h2>
              <motion.p
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: 0.1 }}
                className="text-lg text-slate-500 max-w-2xl mx-auto"
              >
                Our tool is designed from the ground up to understand the nuances of the Indian property market, from BHK layouts to local area dynamics.
              </motion.p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-12">
              <InfoItem
                icon={<Info className="w-5 h-5 text-blue-500" />}
                title="BHK Logic"
                description="Intelligent handling of Indian BHK terminology."
              />
              <InfoItem
                icon={<Mail className="w-5 h-5 text-blue-500" />}
                title="Support"
                description="Dedicated support for developers and users."
              />
              <InfoItem
                icon={<Github className="w-5 h-5 text-blue-500" />}
                title="Open Source"
                description="Completely transparent and open for contributions."
              />
              <InfoItem
                icon={<Heart className="w-5 h-5 text-blue-500" />}
                title="User Focused"
                description="Designed with a focus on simplicity and accuracy."
              />
            </div>
          </div>
        </section>
      </main>

      <footer className="py-12 bg-slate-50 border-t border-slate-200">
        <div className="container mx-auto px-6 lg:px-12 max-w-7xl">
          <div className="flex flex-col md:flex-row justify-between items-center gap-8">
            <div className="flex items-center gap-2 group">
              <div className="p-1.5 bg-blue-500 rounded-lg">
                <Heart className="text-white w-4 h-4 fill-white" />
              </div>
              <span className="font-bold text-lg tracking-tight text-slate-800">
                Prop<span className="text-blue-600">Predict</span>
              </span>
            </div>
            <div className="flex items-center gap-8 text-sm font-medium text-slate-500">
              <a href="#" className="hover:text-blue-600 transition-colors">Privacy Policy</a>
              <a href="#" className="hover:text-blue-600 transition-colors">Terms of Service</a>
              <a href="#" className="hover:text-blue-600 transition-colors">Contact Us</a>
            </div>
            <p className="text-slate-400 text-sm">
              &copy; {new Date().getFullYear()} PropPredict. Built with Scikit-learn & Streamlit.
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}

function InfoItem({ icon, title, description }: { icon: React.ReactNode; title: string; description: string }) {
  return (
    <div className="flex flex-col items-center text-center p-6 hover:bg-slate-50 rounded-2xl transition-colors">
      <div className="w-10 h-10 rounded-full bg-blue-50 flex items-center justify-center mb-4">
        {icon}
      </div>
      <h4 className="text-lg font-bold text-slate-900 mb-2">{title}</h4>
      <p className="text-slate-500 text-sm leading-relaxed">{description}</p>
    </div>
  );
}
